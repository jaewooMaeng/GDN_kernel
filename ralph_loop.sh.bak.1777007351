#!/usr/bin/env bash
# =============================================================================
# ralph_loop.sh — Codex 기반 CUDA 커널 최적화 무한 루프
#
#   종료 조건   : avg latency < ${TARGET_LATENCY_MS} ms
#   보조 목표   : NCU Duration 감소
#   세션 수명   : 30분 동안 로그 갱신 없으면 kill 후 새 세션
#   세션 성공 후: git commit
#   승인 정책   : codex의 모든 approval 자동 수락
# =============================================================================
set -uo pipefail

# ---------- 사용자 설정 (env 로 override 가능) --------------------------------
TARGET_LATENCY_MS="${TARGET_LATENCY_MS:-0.009}"      # 종료 조건 (ms)
IDLE_TIMEOUT_SEC="${IDLE_TIMEOUT_SEC:-1800}"          # 30분
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"          # 로그 mtime 체크 주기
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"              # 안전 상한
MAX_CONSECUTIVE_SESSION_FAILURES="${MAX_CONSECUTIVE_SESSION_FAILURES:-3}"
SESSION_FAILURE_SLEEP_SEC="${SESSION_FAILURE_SLEEP_SEC:-10}"
STATE_DIR="${STATE_DIR:-ralph_state}"
LOGS_DIR="${LOGS_DIR:-ralph_logs}"
WORKFLOW_FILE="${WORKFLOW_FILE:-workflow.md}"
CODEX_BIN="${CODEX_BIN:-codex}"
# codex exec 의 승인/샌드박스 옵션. 구버전 호환 위해 override 가능.
CODEX_FLAGS="${CODEX_FLAGS:---dangerously-bypass-approvals-and-sandbox --skip-git-repo-check}"

DEFAULT_PYTHON_BIN="/opt/homebrew/Caskroom/miniforge/base/envs/fi-bench/bin/python"
DEFAULT_MODAL_BIN="/opt/homebrew/Caskroom/miniforge/base/envs/fi-bench/bin/modal"
[[ -x "$DEFAULT_PYTHON_BIN" ]] || DEFAULT_PYTHON_BIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || printf 'python')"
[[ -x "$DEFAULT_MODAL_BIN" ]] || DEFAULT_MODAL_BIN="$(command -v modal 2>/dev/null || printf 'modal')"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
MODAL_BIN="${MODAL_BIN:-$DEFAULT_MODAL_BIN}"
NCU_WORKLOAD_UUID="${NCU_WORKLOAD_UUID:-eaf0a285-447c-4432-8e68-d287acc3cb08}"
NCU_SET="${NCU_SET:-detailed}"

LATENCY_FILE="$STATE_DIR/latest_latency.txt"          # codex 가 avg latency 를 이 파일에 기록
PROFILE_LOG="${PROFILE_LOG:-log.md}"                  # 직전 iteration 의 ncu 분석 결과 및 누적 메모
NCU_DURATION_FILE="$STATE_DIR/latest_ncu_duration_us.txt"

mkdir -p "$STATE_DIR" "$LOGS_DIR"

# ---------- 유틸 --------------------------------------------------------------
log()  { printf '\033[1;36m[ralph %s]\033[0m %s\n' "$(date '+%F %T')" "$*"; }
warn() { printf '\033[1;33m[ralph %s] WARN:\033[0m %s\n' "$(date '+%F %T')" "$*"; }
die()  { printf '\033[1;31m[ralph %s] FATAL:\033[0m %s\n' "$(date '+%F %T')" "$*"; exit 1; }

# GNU/BSD stat 호환
file_mtime() {
    stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0
}

# line-buffered output 을 보장 (coreutils stdbuf 가 있으면)
USE_STDBUF=0
if command -v stdbuf >/dev/null 2>&1; then
    USE_STDBUF=1
fi

command -v "$CODEX_BIN" >/dev/null 2>&1 || die "codex 바이너리를 찾을 수 없음: $CODEX_BIN"
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

# ---------- 종료조건 체크 -----------------------------------------------------
# LATENCY_FILE 안의 숫자가 TARGET_LATENCY_MS 보다 작으면 0 (= 종료), 아니면 1.
check_termination() {
    [[ -f "$LATENCY_FILE" ]] || return 1
    local latency
    latency="$(tr -d ' \t\r\n' < "$LATENCY_FILE")"
    [[ -n "$latency" ]] || return 1
    # bash 는 부동소수점 비교가 안되니 awk 로
    awk -v l="$latency" -v t="$TARGET_LATENCY_MS" \
        'BEGIN { exit !(l+0 > 0 && l+0 < t+0) }'
}

# ---------- Codex 에 줄 프롬프트 ---------------------------------------------
build_prompt() {
    local iter="$1"
    local pack_cmd="$PYTHON_BIN scripts/pack_solution.py"
    local bench_cmd="$MODAL_BIN run scripts/run_modal.py"
    local ncu_cmd="$MODAL_BIN run scripts/run_ncu_modal.py --workload-uuid $NCU_WORKLOAD_UUID --ncu-set $NCU_SET"
    cat <<EOF
너는 NVIDIA Blackwell (sm_100a, B200) 타깃 CUDA 커널 최적화 전문가다.
최종 목표: GDN (Gated DeltaNet) decode 커널의 avg latency 를 ${TARGET_LATENCY_MS} ms 미만으로 내리는 것.
보조 목표: NCU profiling 결과의 kernel Duration(us)을 줄이는 방향을 우선 고려한다.
단, benchmark avg/median latency 와 correctness 를 최종 판정 기준으로 둔다. NCU Duration 이 줄어도 benchmark latency 가 후퇴하면 유지하지 않는다.

이번은 iteration #${iter} 이다.
아래 4단계를 순서대로 수행하고, 각 단계의 결론을 최종 응답에 요약해서 남겨라.

=========================================================
Step 1. 계획 수립
=========================================================
1) 다음 두 파일을 읽는다.
     - 워크플로 문서:        ${WORKFLOW_FILE}
     - 직전 iteration ncu 로그: ${PROFILE_LOG}
   (이번이 첫 iteration 이라 로그가 없다면 커널 코드 자체만 분석해서 가장 합리적인 첫 수를 두어라.)
2) ${WORKFLOW_FILE} 와 ${PROFILE_LOG} 에서 [시도됨], 후퇴, 재시도 금지, FAILED, bad direction 으로 기록된 항목을 먼저 정리한다.
   이미 나쁘다고 판정된 방향은 같은 이유가 여전히 유효하면 반복하지 않는다. 재시도하려면 "이전 실패 이유가 왜 더 이상 유효하지 않은지"를 명시해야 한다.
3) 프로파일의 bottleneck 지표 (NCU Duration, waves/SM, memory throughput, DRAM/L1/L2 throughput,
   achieved occupancy, issue slots busy, registers/thread, local memory spilling, cache hit rate 등) 를 근거로
   이번 iteration 에 적용할 최적화를 1~3 개 선택한다.
4) 각 후보에 대해 기대 효과 / 구현 난이도 / 리스크 / NCU Duration 감소 가능성을 한국어로 간단히 정리한다.

=========================================================
Step 2. PM 검토
=========================================================
이어서 너는 NVIDIA senior performance engineer PM 에 빙의한다.
PM 은 다음 기준으로 냉정하게 계획을 검토한다.
    (a) 프로파일 근거와 정합하는가
    (b) 구현 범위가 한 iteration 에 넣기에 과하지 않은가
    (c) 다른 최적화와의 상호작용 / 회귀 리스크는 없는가
PM 이 만족하지 못하면 계획을 수정 → 재검토를 반복한다.
PM 이 명시적으로 "APPROVED" 라고 선언한 뒤에만 Step 3 으로 진행한다.

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
측정이 성공했다면 ncu 를 이용해 현재 커널을 프로파일하고 결과를 파일로 저장한다.
NCU 는 README helper 단독 경로가 아니라, 이 repo 에서 검증된 Modal runner/fallback 경로를 사용한다:
    ${ncu_cmd}

성공 판정은 출력에 gdn_decode_kernel profiling 과 NCU metric section 이 실제로 포함되는지로 한다.
README helper 가 "No kernels were profiled" 를 출력해도, fallback 이 gdn_decode_kernel 을 profile 했으면 성공으로 본다.

경로는 ${PROFILE_LOG} 로 통일한다. 기존 로그에 있는 history, 실패 이유, 재시도 금지 메모를 잃지 않도록 append 를 기본으로 한다.
로그 뒷부분에 분석 섹션을 append 한다.
    ## iter #${iter}
    - 적용한 최적화:
    - 측정된 avg latency:
    - NCU Duration:
    - 남아있는 주요 bottleneck:
    - 이번에 시도했거나 검토했지만 안 좋다고 판단한 방향:
      - 후보:
      - 안 좋은 이유:
      - 재시도 가능 조건:
    - 다음 iteration 에서 시도할만한 후보 2~3 개:

NCU 출력의 Duration(us) 숫자만 아래 파일에도 덮어써라:
    printf '%s' "<duration_us>" > ${NCU_DURATION_FILE}

마지막으로 ${WORKFLOW_FILE} 의 history 섹션에 한 줄 요약을 추가한다.
후퇴/롤백/보류한 방향이 있으면 ${WORKFLOW_FILE} 의 후보 목록 또는 성능 로그에도 이유를 함께 남긴다.

=========================================================
실행 규칙
=========================================================
- 필요한 모든 쉘 명령을 네가 직접 실행해라. 어떤 승인 요청이 떠도 모두 YES 로 진행.
- 중요한 결정은 한국어로 간결히 설명한다.
- 단계 경계마다 한두 줄짜리 체크포인트를 콘솔에 남긴다.
- 종료 조건(< ${TARGET_LATENCY_MS} ms)을 이번 iteration 에 이미 달성했다면, 불필요한 추가 변경 없이 마무리한다.
EOF
}

# ---------- Codex 세션 실행 + idle watchdog ----------------------------------
run_codex_session() {
    local iter="$1"
    local iter_dir="$LOGS_DIR/iter_$(printf '%04d' "$iter")"
    mkdir -p "$iter_dir"
    local session_log="$iter_dir/codex.log"
    local prompt_file="$iter_dir/prompt.txt"

    build_prompt "$iter" > "$prompt_file"
    : > "$session_log"
    touch "$session_log"

    log "iter #$iter → codex 세션 시작"
    log "    prompt: $prompt_file"
    log "    log   : $session_log"

    # codex 를 백그라운드로 띄운다. setsid 가 있으면 새 process group 을
    # 만들어 watchdog 이 자손까지 정리할 수 있게 한다. macOS bash 3.2는
    # set -u 상태에서 빈 배열 확장이 터지므로 실행 분기를 직접 나눈다.
    # shellcheck disable=SC2086
    if command -v setsid >/dev/null 2>&1; then
        if (( USE_STDBUF )); then
            setsid stdbuf -oL -eL "$CODEX_BIN" exec $CODEX_FLAGS \
                "$(cat "$prompt_file")" \
                >>"$session_log" 2>&1 &
        else
            setsid "$CODEX_BIN" exec $CODEX_FLAGS \
                "$(cat "$prompt_file")" \
                >>"$session_log" 2>&1 &
        fi
    else
        if (( USE_STDBUF )); then
            stdbuf -oL -eL "$CODEX_BIN" exec $CODEX_FLAGS \
                "$(cat "$prompt_file")" \
                >>"$session_log" 2>&1 &
        else
            "$CODEX_BIN" exec $CODEX_FLAGS \
                "$(cat "$prompt_file")" \
                >>"$session_log" 2>&1 &
        fi
    fi
    local codex_pid=$!

    # watchdog: IDLE_TIMEOUT_SEC 동안 session_log 의 mtime 이 안 바뀌면 죽임
    (
        while kill -0 "$codex_pid" 2>/dev/null; do
            sleep "$POLL_INTERVAL_SEC"
            local mtime now idle
            mtime=$(file_mtime "$session_log")
            now=$(date +%s)
            idle=$(( now - mtime ))
            if (( idle > IDLE_TIMEOUT_SEC )); then
                echo "" >> "$session_log"
                echo "[watchdog] ${idle}s 동안 로그 갱신 없음 → codex 세션 강제 종료" \
                    | tee -a "$session_log"
                # process group 째로 정리
                if kill -0 -- "-$codex_pid" 2>/dev/null; then
                    kill -TERM -- "-$codex_pid" 2>/dev/null || true
                    sleep 5
                    kill -KILL -- "-$codex_pid" 2>/dev/null || true
                else
                    pkill -TERM -P "$codex_pid" 2>/dev/null || true
                    kill  -TERM    "$codex_pid" 2>/dev/null || true
                    sleep 5
                    pkill -KILL -P "$codex_pid" 2>/dev/null || true
                    kill  -KILL    "$codex_pid" 2>/dev/null || true
                fi
                exit 0
            fi
        done
    ) &
    local watchdog_pid=$!

    # 실시간 tail (사용자가 터미널에서 진행상황을 볼 수 있도록)
    tail -n +1 -f "$session_log" &
    local tail_pid=$!

    wait "$codex_pid"
    local rc=$?

    # watchdog / tail 정리
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
    sleep 1
    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true

    log "iter #$iter → codex 세션 종료 (exit=$rc)"
    return "$rc"
}

# ---------- git commit --------------------------------------------------------
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
    msg="ralph iter $(printf '%04d' "$iter")"
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

# ---------- 정리 핸들러 -------------------------------------------------------
cleanup() {
    log "정리 중..."
    # 현재 쉘의 자식 job 들 전부 종료
    local pids
    pids="$(jobs -p)"
    [[ -n "$pids" ]] && kill $pids 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'log "SIGINT 수신 → 종료"; exit 130' INT
trap 'log "SIGTERM 수신 → 종료"; exit 143' TERM

# ---------- 메인 루프 ---------------------------------------------------------
log "ralph loop 시작"
log "    target       : avg_latency < ${TARGET_LATENCY_MS} ms"
log "    workflow     : ${WORKFLOW_FILE}"
log "    ncu log      : ${PROFILE_LOG}"
log "    ncu command  : ${MODAL_BIN} run scripts/run_ncu_modal.py --workload-uuid ${NCU_WORKLOAD_UUID} --ncu-set ${NCU_SET}"
log "    idle timeout : ${IDLE_TIMEOUT_SEC} s"
log "    max iters    : ${MAX_ITERATIONS}"
log "    max failures : ${MAX_CONSECUTIVE_SESSION_FAILURES}"

iter=0
consecutive_session_failures=0
while (( iter < MAX_ITERATIONS )); do
    iter=$(( iter + 1 ))

    if check_termination; then
        log "🎉 종료조건 달성! latency=$(cat "$LATENCY_FILE")ms < ${TARGET_LATENCY_MS}ms"
        exit 0
    fi

    if run_codex_session "$iter"; then
        consecutive_session_failures=0
        commit_iteration "$iter"
    else
        consecutive_session_failures=$(( consecutive_session_failures + 1 ))
        warn "iter #$iter → codex 세션이 비정상 종료되어 commit skip"
        if (( consecutive_session_failures >= MAX_CONSECUTIVE_SESSION_FAILURES )); then
            die "codex 세션이 ${consecutive_session_failures}회 연속 실패하여 루프를 중단"
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
