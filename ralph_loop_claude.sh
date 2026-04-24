#!/usr/bin/env bash
# =============================================================================
# ralph_loop_claude.sh — Claude Code(`claude -p`) 기반 CUDA 커널 최적화 무한 루프
#
#   종료 조건   : avg latency < ${TARGET_LATENCY_MS} ms
#   보조 목표   : NCU Duration 감소
#   iteration   : (1) plan+PM 전용 Claude → plan.md 에 APPROVED
#                 (2) 구현+bench+NCU (단일 인스턴스 락, claude 로그는 짧게 유지)
#   세션 수명   : 30분 동안 로그 갱신 없으면 kill 후 새 세션
#   세션 성공 후: git commit
#   승인 정책   : --dangerously-skip-permissions (env 의 CLAUDE_FLAGS 로 override)
# =============================================================================
set -uo pipefail

# ---------- 사용자 설정 (env 로 override 가능) --------------------------------
TARGET_LATENCY_MS="${TARGET_LATENCY_MS:-0.009}"      # 종료 조건 (ms)
IDLE_TIMEOUT_SEC="${IDLE_TIMEOUT_SEC:-1800}"          # 30분
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"          # 로그 mtime 체크 주기
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"              # 안전 상한
MAX_CONSECUTIVE_SESSION_FAILURES="${MAX_CONSECUTIVE_SESSION_FAILURES:-3}"
SESSION_FAILURE_SLEEP_SEC="${SESSION_FAILURE_SLEEP_SEC:-10}"
MAX_PLAN_RETRIES="${MAX_PLAN_RETRIES:-3}"
# codex 루프와 병행 시 겹치지 않게 기본값을 분리
STATE_DIR="${STATE_DIR:-ralph_state_claude}"
LOGS_DIR="${LOGS_DIR:-ralph_logs_claude}"
# per-phase claude 로그(ralph_logs_claude/iter_*/claude_*.log)이 이 바이트를 넘기면 끝부분만 유지
CLAUDE_LOG_MAX_BYTES="${CLAUDE_LOG_MAX_BYTES:-65536}"
WORKFLOW_FILE="${WORKFLOW_FILE:-workflow.md}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
# Codex 루프의 --dangerously-bypass-... 에 대응: 권한 프롬프트 스킵 (필요 시 --bare 등 추가)
CLAUDE_FLAGS="${CLAUDE_FLAGS:---dangerously-skip-permissions}"

DEFAULT_PYTHON_BIN="/opt/homebrew/Caskroom/miniforge/base/envs/fi-bench/bin/python"
DEFAULT_MODAL_BIN="/opt/homebrew/Caskroom/miniforge/base/envs/fi-bench/bin/modal"
[[ -x "$DEFAULT_PYTHON_BIN" ]] || DEFAULT_PYTHON_BIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || printf 'python')"
[[ -x "$DEFAULT_MODAL_BIN" ]] || DEFAULT_MODAL_BIN="$(command -v modal 2>/dev/null || printf 'modal')"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
MODAL_BIN="${MODAL_BIN:-$DEFAULT_MODAL_BIN}"
NCU_WORKLOAD_UUID="${NCU_WORKLOAD_UUID:-eaf0a285-447c-4432-8e68-d287acc3cb08}"
NCU_SET="${NCU_SET:-detailed}"

LATENCY_FILE="$STATE_DIR/latest_latency.txt"          # 에이전트가 avg latency 를 기록
PROFILE_LOG="${PROFILE_LOG:-log.md}"                  # NCU/분석 (append·장문 OK; 토큰 절약은 state/logs 쪽)
PLAN_FILE="$STATE_DIR/plan.md"                        # plan+PM 단계 산출 (문자열 APPROVED 필수)
NCU_DURATION_FILE="$STATE_DIR/latest_ncu_duration_us.txt"
ITER_METRICS_FILE="$STATE_DIR/iter_metrics.tsv"
RALPH_LOCK_DIR="$STATE_DIR/ralph_loop_claude_lock"
RALPH_LOCK_HELD=0

mkdir -p "$STATE_DIR" "$LOGS_DIR"

# ---------- 유틸 --------------------------------------------------------------
log()  { printf '\033[1;36m[ralph-claude %s]\033[0m %s\n' "$(date '+%F %T')" "$*"; }
warn() { printf '\033[1;33m[ralph-claude %s] WARN:\033[0m %s\n' "$(date '+%F %T')" "$*"; }
die()  { printf '\033[1;31m[ralph-claude %s] FATAL:\033[0m %s\n' "$(date '+%F %T')" "$*"; exit 1; }

file_mtime() {
    stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0
}

USE_STDBUF=0
if command -v stdbuf >/dev/null 2>&1; then
    USE_STDBUF=1
fi

acquire_lock() {
    mkdir -p "$STATE_DIR" || die "STATE_DIR 생성 실패"
    if ! mkdir "$RALPH_LOCK_DIR" 2>/dev/null; then
        die "락 획득 실패: 다른 ralph_loop_claude 가 실행 중 ($RALPH_LOCK_DIR)"
    fi
    RALPH_LOCK_HELD=1
}

release_lock() {
    [[ "$RALPH_LOCK_HELD" = 1 ]] || return 0
    rmdir "$RALPH_LOCK_DIR" 2>/dev/null || true
    RALPH_LOCK_HELD=0
}

command -v "$CLAUDE_BIN" >/dev/null 2>&1 || die "claude 바이너리를 찾을 수 없음: $CLAUDE_BIN"
command -v git         >/dev/null 2>&1 || warn "git 이 없음 — commit 단계가 skip 됨"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || warn "python 실행 파일 확인 실패: $PYTHON_BIN"
command -v "$MODAL_BIN"  >/dev/null 2>&1 || warn "modal 실행 파일 확인 실패: $MODAL_BIN"
[[ -f "$WORKFLOW_FILE" ]] || die "워크플로 문서를 찾을 수 없음: $WORKFLOW_FILE"
[[ -f scripts/run_ncu_modal.py ]] || die "NCU Modal runner를 찾을 수 없음: scripts/run_ncu_modal.py"
if [[ ! -f "$PROFILE_LOG" ]]; then
    warn "NCU log 파일이 없어 새로 생성: $PROFILE_LOG"
    printf '# NCU Profiling Log\n\n' > "$PROFILE_LOG"
fi
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if ! git diff --quiet || ! git diff --cached --quiet || \
       [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
        warn "현재 worktree 에 기존 변경사항이 있음. 첫 성공 commit 에 함께 포함될 수 있음."
    fi
fi

# ---------- 종료조건 ---------------------------------------------------------
check_termination() {
    [[ -f "$LATENCY_FILE" ]] || return 1
    local latency
    latency="$(tr -d ' \t\r\n' < "$LATENCY_FILE")"
    [[ -n "$latency" ]] || return 1
    awk -v l="$latency" -v t="$TARGET_LATENCY_MS" \
        'BEGIN { exit !(l+0 > 0 && l+0 < t+0) }'
}

check_plan_approved() {
    [[ -f "$PLAN_FILE" ]] || return 1
    grep -qi 'APPROVED' "$PLAN_FILE"
}

compact_claude_log_file() {
    local f="$1"
    local maxb="${CLAUDE_LOG_MAX_BYTES:-65536}"
    [[ -f "$f" ]] || return 0
    local sz
    sz=$(wc -c < "$f" 2>/dev/null | tr -d ' ' || echo 0)
    [[ -n "$sz" ]] || sz=0
    (( sz > maxb )) || return 0
    if command -v tail >/dev/null 2>&1; then
        tail -c "$maxb" "$f" > "${f}.$$.compact" && mv "${f}.$$.compact" "$f"
    fi
}

# ---------- Claude 프롬프트 (phase: plan | impl) --------------------------------
build_prompt_plan() {
    local iter="$1"
    local pack_cmd="$PYTHON_BIN scripts/pack_solution.py"
    local bench_cmd="$MODAL_BIN run scripts/run_modal.py"
    local ncu_cmd="$MODAL_BIN run scripts/run_ncu_modal.py --workload-uuid $NCU_WORKLOAD_UUID --ncu-set $NCU_SET"
    cat <<EOF
너는 NVIDIA Blackwell (sm_100a, B200) 타깃 CUDA 커널 최적화 전문가다.
최종 목표: GDN (Gated DeltaNet) decode 커널의 avg latency 를 ${TARGET_LATENCY_MS} ms 미만으로 내리는 것.
보조 목표: NCU profiling 결과의 kernel Duration(us)을 줄이는 방향을 우선 고려한다.
단, benchmark avg/median latency 와 correctness 를 최종 판정 기준으로 둔다.

**이번 세션은 iteration #${iter} 의 Step 1~2 전용이다.**
- **커널을 수정하거나, pack/bench, NCU(Modal) 명령을 실행하지 말라.** (아래 ${pack_cmd} / ${bench_cmd} / ${ncu_cmd} 는 **실행 금지**)
- ${PROFILE_LOG} 나 ${WORKFLOW_FILE} 는 읽기용으로 사용한다.

이번에 반드시 생성·덮어써야 할 파일 (경로를 그대로 쓰라):
    ${PLAN_FILE}

${PLAN_FILE} 에 다음을 **한국어**로 담는다: Step 1 정리(후보, 리스크), Step 2 PM 검토 대화/결론.
PM 은 (a) 프로파일·근거 (b) 한 iteration에 과한지 (c) 회귀 리스크 를 검토한다. 불만이면 계획을 수정·재검토한다.
PM 이 최종 승인하면 본문 어딘가에 **대문자** 로 **APPROVED** 가 반드시 나타나게 쓴다(줄이 아니면 단어로 표기).

최종 응답(콘솔)에는 Step1·2 요약과 ${PLAN_FILE} 에 기록 완료를 짧게 알린다.
EOF
}

build_prompt_impl() {
    local iter="$1"
    local pack_cmd="$PYTHON_BIN scripts/pack_solution.py"
    local bench_cmd="$MODAL_BIN run scripts/run_modal.py"
    local ncu_cmd="$MODAL_BIN run scripts/run_ncu_modal.py --workload-uuid $NCU_WORKLOAD_UUID --ncu-set $NCU_SET"
    cat <<EOF
너는 NVIDIA Blackwell (sm_100a, B200) 타깃 CUDA 커널 최적화 전문가다.
최종 목표: GDN (Gated DeltaNet) decode 커널의 avg latency 를 ${TARGET_LATENCY_MS} ms 미만으로 내리는 것.
보조 목표: NCU kernel Duration(us) 감소. 최종 판정은 benchmark latency 와 correctness.

이번은 iteration #${iter} 이다. **Step 1~2(plan/PM) 는 이미 끝났다.**
- 반드시 먼저 읽는다: ${PLAN_FILE}
- ${PLAN_FILE} 에 **APPROVED** 가 없다면, **solution/cuda/kernel.cu 를 수정하지 말고** 벤치/ncu도 스킵하고 이유를 설명하라.
- **APPROVED** 가 있을 때에만 Step 3~4 를 수행한다.

${PROFILE_LOG} 에는 분석·NCU 인용·장문을 **자유롭게** append 해도 좋다. (누적 history·실패 메모 유지를 위해 append 권장)
${STATE_DIR}/ 와 ${LOGS_DIR}/ 는 스크립트가 짧은 숫자/메트릭만 유지—너는 ${LATENCY_FILE}·${NCU_DURATION_FILE} 덮어쓰기와 위 프로파일 로그만 갱신하면 된다.

=========================================================
Step 3. 구현 및 성능 측정
=========================================================
승인된 계획대로 CUDA 커널 코드를 직접 수정한다.
수정이 끝나면 아래 명령을 실행한다 (셀프 승인):
    ${pack_cmd}
    ${bench_cmd}

${WORKFLOW_FILE} 가 Phase 3+ 반복 측정을 요구하면, 같은 benchmark 명령을 필요한 횟수만큼 반복 실행하고 median 을 판정에 사용한다.

위 두 명령의 stdout 을 그대로 출력에 포함시켜라.
측정된 avg latency 또는 median latency (단위 ms) 숫자만 아래 파일에 덮어써라:
    printf '%s' "<avg_latency_ms>" > ${LATENCY_FILE}

빌드 또는 실행이 실패하면 원인을 분석해서 최대 2 회까지 수정-재실행을 시도한다.
그래도 실패하면 이번 세션의 커널 변경을 되돌리고(revert),
실패 원인을 ${PROFILE_LOG} 끝에 [FAILED iter #${iter}] 섹션으로 append 한다.

=========================================================
Step 4. 프로파일링 & 다음 iteration 용 로그
=========================================================
측정이 성공했다면 ncu 를 이용해 현재 커널을 프로파일한다.
    ${ncu_cmd}

성공 판정은 출력에 gdn_decode_kernel profiling 과 NCU metric section 이 실제로 포함되는지로 한다.
README helper 가 "No kernels were profiled" 를 출력해도, fallback 이 gdn_decode_kernel 을 profile 했으면 성공으로 본다.

${PROFILE_LOG} 뒤에 ## iter #${iter} 섹션 등으로 분석을 append 한다(형식은 기존 ralph 루프와 동일해도 됨).

NCU 출력의 Duration(us) 숫자만 아래 파일에도 덮어써라:
    printf '%s' "<duration_us>" > ${NCU_DURATION_FILE}

마지막으로 ${WORKFLOW_FILE} 의 history 섹션에 한 줄 요약을 추가한다.
후퇴/롤백/보류한 방향이 있으면 ${WORKFLOW_FILE} 의 후보 목록 또는 성능 로그에도 이유를 함께 남긴다.

=========================================================
실행 규칙
=========================================================
- 필요한 모든 쉘 명령을 네가 직접 실행해라. 권한/도구 승인이 필요하면 --dangerously-skip-permissions 가 이미 켜져 있어야 한다(아니면 스크립트의 CLAUDE_FLAGS 를 조정).
- 중요한 결정은 한국어로 간결히 설명한다.
- 단계 경계마다 한두 줄짜리 체크포인트를 콘솔에 남긴다.
- 종료 조건(< ${TARGET_LATENCY_MS} ms)을 이번 iteration 에 이미 달성했다면, 불필요한 추가 변경 없이 마무리한다.
EOF
}

# ---------- Claude Code (`claude -p`) 세션 + idle watchdog (세션 끝에 로그 tail 압축) --
run_claude_phase() {
    local iter="$1"
    local phase="$2"
    local iter_dir="$LOGS_DIR/iter_$(printf '%04d' "$iter")"
    mkdir -p "$iter_dir"
    local session_log="$iter_dir/claude_${phase}.log"
    local prompt_file="$iter_dir/prompt_${phase}.txt"

    case "$phase" in
        plan) build_prompt_plan "$iter" > "$prompt_file" ;;
        impl) build_prompt_impl "$iter"  > "$prompt_file" ;;
        *) die "run_claude_phase: 잘못된 phase: $phase" ;;
    esac
    : > "$session_log"
    touch "$session_log"

    log "iter #$iter → claude -p 세션 시작 ($phase)"
    log "    prompt: $prompt_file"
    log "    log   : $session_log"

    # shellcheck disable=SC2086
    if command -v setsid >/dev/null 2>&1; then
        if (( USE_STDBUF )); then
            setsid stdbuf -oL -eL "$CLAUDE_BIN" -p \
                "$(cat "$prompt_file")" \
                $CLAUDE_FLAGS \
                >>"$session_log" 2>&1 &
        else
            setsid "$CLAUDE_BIN" -p \
                "$(cat "$prompt_file")" \
                $CLAUDE_FLAGS \
                >>"$session_log" 2>&1 &
        fi
    else
        if (( USE_STDBUF )); then
            stdbuf -oL -eL "$CLAUDE_BIN" -p \
                "$(cat "$prompt_file")" \
                $CLAUDE_FLAGS \
                >>"$session_log" 2>&1 &
        else
            "$CLAUDE_BIN" -p \
                "$(cat "$prompt_file")" \
                $CLAUDE_FLAGS \
                >>"$session_log" 2>&1 &
        fi
    fi
    local claude_pid=$!

    (
        while kill -0 "$claude_pid" 2>/dev/null; do
            sleep "$POLL_INTERVAL_SEC"
            local mtime now idle
            mtime=$(file_mtime "$session_log")
            now=$(date +%s)
            idle=$(( now - mtime ))
            if (( idle > IDLE_TIMEOUT_SEC )); then
                echo "" >> "$session_log"
                echo "[watchdog] ${idle}s 동안 로그 갱신 없음 → claude -p 세션 강제 종료" \
                    | tee -a "$session_log"
                if kill -0 -- "-$claude_pid" 2>/dev/null; then
                    kill -TERM -- "-$claude_pid" 2>/dev/null || true
                    sleep 5
                    kill -KILL -- "-$claude_pid" 2>/dev/null || true
                else
                    pkill -TERM -P "$claude_pid" 2>/dev/null || true
                    kill  -TERM    "$claude_pid" 2>/dev/null || true
                    sleep 5
                    pkill -KILL -P "$claude_pid" 2>/dev/null || true
                    kill  -KILL    "$claude_pid" 2>/dev/null || true
                fi
                exit 0
            fi
        done
    ) &
    local watchdog_pid=$!

    tail -n +1 -f "$session_log" &
    local tail_pid=$!

    wait "$claude_pid"
    local rc=$?

    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
    sleep 1
    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true

    compact_claude_log_file "$session_log"
    log "iter #$iter → claude 세션 종료 $phase (exit=$rc)"
    return "$rc"
}

# 루프당 한 줄: iter, 시각, latency, ncu, status (ok / plan_* / impl_*)
append_iter_metrics() {
    local iter="$1"
    local status="$2"
    local ts
    ts=$(date '+%Y-%m-%dT%H:%M:%S%z' 2>/dev/null || date '+%F %T')
    local lat="-"
    local ncu="-"
    if [[ "$status" = "ok" ]]; then
        [[ -f "$LATENCY_FILE" ]] && lat="$(tr -d ' \t\r\n' < "$LATENCY_FILE")"
        [[ -f "$NCU_DURATION_FILE" ]] && ncu="$(tr -d ' \t\r\n' < "$NCU_DURATION_FILE")"
    elif [[ "$status" = "impl_session_fail" ]]; then
        [[ -f "$LATENCY_FILE" ]] && lat="$(tr -d ' \t\r\n' < "$LATENCY_FILE")"
        [[ -f "$NCU_DURATION_FILE" ]] && ncu="$(tr -d ' \t\r\n' < "$NCU_DURATION_FILE")"
    fi
    if [[ ! -f "$ITER_METRICS_FILE" ]]; then
        printf 'iter\ttimestamp\tlatency_ms\tncu_us\tstatus\n' > "$ITER_METRICS_FILE"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$iter" "$ts" "$lat" "$ncu" "$status" >> "$ITER_METRICS_FILE"
}

write_kernel_sha_state() {
    local k="solution/cuda/kernel.cu"
    [[ -f "$k" ]] || return 0
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$k" | awk '{print $1}' > "$STATE_DIR/last_kernel_sha.txt"
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$k" | awk '{print $1}' > "$STATE_DIR/last_kernel_sha.txt"
    fi
}

commit_iteration() {
    local iter="$1"
    command -v git >/dev/null 2>&1 || return 0
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
        warn "git repo 가 아님 → commit skip"
        return 0
    }

    if git diff --quiet && git diff --cached --quiet && \
       [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
        log "iter #$iter → 변경사항 없음, commit skip"
        return 0
    fi

    git add -A
    local msg
    msg="ralph-claude iter $(printf '%04d' "$iter")"
    if [[ -f "$LATENCY_FILE" ]]; then
        local lat
        lat="$(tr -d ' \t\r\n' < "$LATENCY_FILE")"
        [[ -n "$lat" ]] && msg+=" — avg_latency=${lat}ms"
    fi
    if git commit -m "$msg" >/dev/null 2>&1; then
        log "iter #$iter → git commit: $msg"
    else
        warn "iter #$iter → git commit 실패 (무시)"
    fi
}

cleanup() {
    log "정리 중..."
    local pids
    pids="$(jobs -p)"
    [[ -n "$pids" ]] && kill $pids 2>/dev/null || true
    wait 2>/dev/null || true
    release_lock
}
trap cleanup EXIT
trap 'log "SIGINT 수신 → 종료"; exit 130' INT
trap 'log "SIGTERM 수신 → 종료"; exit 143' TERM

# 루트 검증 끝난 뒤 단일 인스턴스 락(중복 ralph_loop_claude 방지)
acquire_lock
log "ralph loop (Claude) 시작"
log "    target       : avg_latency < ${TARGET_LATENCY_MS} ms"
log "    workflow     : ${WORKFLOW_FILE}"
log "    plan file    : ${PLAN_FILE}"
log "    ncu log      : ${PROFILE_LOG}"
log "    metrics tsv  : ${ITER_METRICS_FILE}"
log "    ncu command  : ${MODAL_BIN} run scripts/run_ncu_modal.py --workload-uuid ${NCU_WORKLOAD_UUID} --ncu-set ${NCU_SET}"
log "    idle timeout : ${IDLE_TIMEOUT_SEC} s"
log "    max iters    : ${MAX_ITERATIONS}"
log "    plan retries : ${MAX_PLAN_RETRIES}"
log "    max failures : ${MAX_CONSECUTIVE_SESSION_FAILURES}"
log "    claude log   : keep last ${CLAUDE_LOG_MAX_BYTES} bytes per phase"
log "    claude cmd   : ${CLAUDE_BIN} -p <prompt> <CLAUDE_FLAGS...>"

iter=0
consecutive_session_failures=0
while (( iter < MAX_ITERATIONS )); do
    iter=$(( iter + 1 ))
    write_kernel_sha_state

    if check_termination; then
        log "🎉 종료조건 달성! latency=$(cat "$LATENCY_FILE")ms < ${TARGET_LATENCY_MS}ms"
        exit 0
    fi

    last_plan_rc=0
    plan_got_approved=0
    attempt=0
    while (( attempt < MAX_PLAN_RETRIES )); do
        attempt=$(( attempt + 1 ))
        log "iter #$iter plan/PM (시도 ${attempt}/${MAX_PLAN_RETRIES})"
        if run_claude_phase "$iter" plan; then
            last_plan_rc=0
        else
            last_plan_rc=$?
        fi
        if (( last_plan_rc == 0 )) && check_plan_approved; then
            plan_got_approved=1
            break
        fi
        if (( last_plan_rc != 0 )); then
            warn "iter #$iter → plan claude exit=$last_plan_rc"
        else
            warn "iter #$iter → ${PLAN_FILE} 에 APPROVED 없음"
        fi
    done

    if (( plan_got_approved == 0 )); then
        if (( last_plan_rc != 0 )); then
            append_iter_metrics "$iter" "plan_session_fail"
        else
            append_iter_metrics "$iter" "plan_not_approved"
        fi
        consecutive_session_failures=$(( consecutive_session_failures + 1 ))
        warn "iter #$iter → plan 단계를 통과하지 못해 impl 생략 (연속 실패 $consecutive_session_failures)"
        if (( consecutive_session_failures >= MAX_CONSECUTIVE_SESSION_FAILURES )); then
            die "plan 단계를 ${consecutive_session_failures}회 연속 실패하여 루프를 중단"
        fi
        sleep "$SESSION_FAILURE_SLEEP_SEC"
        continue
    fi

    if run_claude_phase "$iter" impl; then
        consecutive_session_failures=0
        if command -v git >/dev/null 2>&1 && \
           [[ -f solution/cuda/kernel.cu ]] && \
           git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            if git diff --quiet -- solution/cuda/kernel.cu 2>/dev/null; then
                warn "iter #$iter → kernel.cu 워크트리 diff 없음 (이번 impl 에서 커널이 안 바뀌었을 수 있음)"
            fi
        fi
        commit_iteration "$iter"
        append_iter_metrics "$iter" "ok"
    else
        consecutive_session_failures=$(( consecutive_session_failures + 1 ))
        warn "iter #$iter → impl claude 실패, commit skip (연속 실패 $consecutive_session_failures)"
        append_iter_metrics "$iter" "impl_session_fail"
        if (( consecutive_session_failures >= MAX_CONSECUTIVE_SESSION_FAILURES )); then
            die "impl 단계를 ${consecutive_session_failures}회 연속 실패하여 루프를 중단"
        fi
        sleep "$SESSION_FAILURE_SLEEP_SEC"
    fi

    if check_termination; then
        log "🎉 종료조건 달성! latency=$(cat "$LATENCY_FILE")ms < ${TARGET_LATENCY_MS}ms"
        exit 0
    fi
done

warn "MAX_ITERATIONS(${MAX_ITERATIONS}) 도달, 루프 종료"
exit 1
