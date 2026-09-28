import { useEffect, useId, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  FileCheck2,
  Info,
  ListChecks,
  LoaderCircle,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

import {
  ApiError,
  companyAnalysisSnapshotKey,
  createCompanyAnalysis,
} from "./api";
import { friendlyBizScoreRule } from "./friendlyEvidence";
import { formatCompanyAge } from "./formatters";
import type {
  AnalysisFindingTopic,
  AnalysisLimitationCode,
  CompanyAnalysisLLMInput,
  CompanyAnalysisResponse,
  CompanyBizScoreResponse,
  VerificationPriority,
} from "./types";

type AnalysisViewState =
  | { kind: "idle" }
  | { kind: "loading" }
  | {
      kind: "success";
      response: CompanyAnalysisResponse;
      refreshing: boolean;
      refreshError?: ApiError;
    }
  | { kind: "error"; error: ApiError };

interface AIAnalysisPanelProps {
  response: CompanyBizScoreResponse;
}

interface EvidenceRow {
  path: string;
  label: string;
  value: string;
}

const TOPIC_LABELS: Record<AnalysisFindingTopic, string> = {
  registration_status: "登記狀態",
  company_age: "成立時間",
  registered_capital_scale: "登記資本",
  registration_change_recency: "登記異動",
  peer_relative_position: "同業相對位置",
  data_completeness: "資料完整度",
};

const LIMITATION_LABELS: Record<AnalysisLimitationCode, string> = {
  public_data_only: "僅限公開資料",
  ai_generated: "AI 產生內容",
  partial_source_data: "來源資料不完整",
  missing_dimension: "部分構面缺值",
  provisional_score: "暫定分數",
  no_numeric_score: "不顯示數字總分",
  benchmark_unavailable: "同業基準不可用",
};

const EVIDENCE_LABELS: Record<string, string> = {
  "/company/tax_id": "統一編號",
  "/company/name": "公司名稱",
  "/company/status/code": "登記狀態代碼",
  "/company/status/description": "登記狀態",
  "/company/capital/registered": "登記資本額",
  "/company/capital/paid_in": "實收資本額",
  "/company/established_at": "核准設立日期",
  "/company/company_age_years": "成立年資",
  "/company/last_changed_at": "最後登記異動日",
  "/company/registration_authority": "登記機關",
  "/bizscore/score": "BizScore",
  "/bizscore/band": "公開資料觀察區間",
  "/bizscore/coverage": "可計分資料覆蓋率",
  "/bizscore/provisional": "是否為暫定結果",
  "/bizscore/as_of": "BizScore 資料基準日",
  "/bizscore/peer_benchmark/peer_index": "同業綜合相對位置",
  "/bizscore/peer_benchmark/age_percentile": "成立年資同業位置",
  "/bizscore/peer_benchmark/capital_percentile": "登記資本同業位置",
  "/bizscore/peer_benchmark/sample_size": "有效同業樣本",
  "/bizscore/benchmark/snapshot_version": "同業基準快照",
  "/source_meta/source": "資料來源",
  "/source_meta/provider": "資料提供機關",
  "/source_meta/fetched_at": "公司資料快照時間",
  "/source_meta/partial": "是否為部分資料",
  "/source_meta/data_freshness": "資料即時狀態",
  "/source_meta/fallback_reason": "即時資料不可用原因",
};

export default function AIAnalysisPanel({ response }: AIAnalysisPanelProps) {
  const taxId = response.data.company.tax_id;
  const snapshotKey = companyAnalysisSnapshotKey(response);
  const [view, setView] = useState<AnalysisViewState>({ kind: "idle" });
  const requestController = useRef<AbortController | null>(null);
  const currentTaxId = useRef(taxId);
  const currentSnapshotKey = useRef(snapshotKey);
  currentTaxId.current = taxId;
  currentSnapshotKey.current = snapshotKey;
  const titleId = useId();
  const descriptionId = useId();
  const actionDescriptionId = useId();

  useEffect(() => {
    requestController.current?.abort();
    requestController.current = null;
    setView({ kind: "idle" });
  }, [snapshotKey, taxId]);

  useEffect(() => {
    return () => requestController.current?.abort();
  }, []);

  const generate = async () => {
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    const previousResponse = view.kind === "success" ? view.response : null;
    if (previousResponse) {
      setView({
        kind: "success",
        response: previousResponse,
        refreshing: true,
      });
    } else {
      setView({ kind: "loading" });
    }

    try {
      const analysisResponse = await createCompanyAnalysis(
        response,
        controller.signal,
      );
      if (
        controller.signal.aborted ||
        currentTaxId.current !== taxId ||
        currentSnapshotKey.current !== snapshotKey
      ) {
        return;
      }
      setView({ kind: "success", response: analysisResponse, refreshing: false });
    } catch (error) {
      if (
        controller.signal.aborted ||
        currentTaxId.current !== taxId ||
        currentSnapshotKey.current !== snapshotKey
      ) {
        return;
      }
      const publicError =
        error instanceof ApiError
          ? error
          : new ApiError(0, {
              code: "AI_ANALYSIS_UNEXPECTED_ERROR",
              message: "產生分析時發生非預期錯誤，請稍後再試。",
              retryable: true,
            });
      if (previousResponse) {
        setView({
          kind: "success",
          response: previousResponse,
          refreshing: false,
          refreshError: publicError,
        });
      } else {
        setView({ kind: "error", error: publicError });
      }
    }
  };

  const isBusy =
    view.kind === "loading" || (view.kind === "success" && view.refreshing);

  return (
    <section
      className={`ai-analysis-section ${view.kind}`}
      aria-labelledby={titleId}
      aria-busy={isBusy}
    >
      <header className="ai-analysis-heading">
        <div className="ai-heading-lockup">
          <span className="ai-heading-icon" aria-hidden="true">
            <Sparkles size={20} />
          </span>
          <div>
            <p className="section-kicker">AI 輔助解讀</p>
            <h3 id={titleId}>合作前查核報告</h3>
          </div>
        </div>
        <span className="ai-content-label">
          <Sparkles aria-hidden="true" size={14} />
          AI 產生內容
        </span>
      </header>

      <div className="ai-score-boundary">
        <ShieldCheck aria-hidden="true" size={18} />
        <p>
          <strong>分數由固定規則產生。</strong>
          上方「為什麼是這個分數」沿用 BizScore v1 既有結果；AI 只整理資料觀察與查核問題，不會重算、加分或扣分。
          <a href="#bizscore-explanation">查看固定規則計分說明</a>
        </p>
      </div>

      {view.kind === "idle" && (
        <div className="ai-analysis-intro">
          <p id={descriptionId} className="ai-intro-copy">
            AI 只會解釋本次取得的公司登記資料、BizScore 與同業基準，整理成可核對的觀察與合作前查核問題。
          </p>
          <div className="ai-preview-grid" aria-label="報告包含內容">
            <div>
              <FileCheck2 aria-hidden="true" size={19} />
              <span>
                <strong>資料觀察</strong>
                <small>每一項都附可展開的公開資料依據</small>
              </span>
            </div>
            <div>
              <ListChecks aria-hidden="true" size={19} />
              <span>
                <strong>查核清單</strong>
                <small>列出簽約與付款前可自行確認的問題</small>
              </span>
            </div>
            <div>
              <ShieldAlert aria-hidden="true" size={19} />
              <span>
                <strong>使用限制</strong>
                <small>明確標示資料缺漏與 AI 內容邊界</small>
              </span>
            </div>
          </div>
        </div>
      )}

      {view.kind === "loading" && <AnalysisLoading />}

      {view.kind === "error" && (
        <AnalysisError error={view.error} />
      )}

      {view.kind === "success" && (
        <>
          {!view.refreshing && !view.refreshError && (
            <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">
              AI 分析完成：{view.response.data.analysis.headline}
            </p>
          )}
          <AnalysisReport response={view.response} />
          {view.refreshError && (
            <AnalysisError error={view.refreshError} preservedReport />
          )}
        </>
      )}

      <AnalysisAction
        view={view}
        onGenerate={() => void generate()}
        descriptionId={actionDescriptionId}
      />
    </section>
  );
}

function AnalysisLoading() {
  return (
    <div className="ai-analysis-loading" role="status" aria-live="polite">
      <LoaderCircle className="spin" aria-hidden="true" size={28} />
      <div>
        <h4>正在產生合作前查核報告</h4>
        <p>系統正在核對已驗證資料、整理觀察並建立查核問題，通常需要數十秒。</p>
      </div>
    </div>
  );
}

function AnalysisError({
  error,
  preservedReport = false,
}: {
  error: ApiError;
  preservedReport?: boolean;
}) {
  const presentation = analysisErrorPresentation(error);
  return (
    <div
      className={`ai-analysis-error${preservedReport ? " preserved" : ""}`}
      role="alert"
    >
      <span className="ai-error-icon" aria-hidden="true">
        <AlertTriangle size={22} />
      </span>
      <div>
        <p className="ai-error-code">
          {preservedReport ? "更新未完成，已保留原報告" : "分析未完成"} · {error.code}
        </p>
        <h4>{presentation.title}</h4>
        <p>{error.message}</p>
        <p className="ai-error-helper">
          {preservedReport
            ? `上一份通過檢查的報告仍保留。${presentation.helper}`
            : presentation.helper}
        </p>
      </div>
    </div>
  );
}

function AnalysisAction({
  view,
  onGenerate,
  descriptionId,
}: {
  view: AnalysisViewState;
  onGenerate: () => void;
  descriptionId: string;
}) {
  const isInitialLoading = view.kind === "loading";
  const isRefreshing = view.kind === "success" && view.refreshing;
  const isBusy = isInitialLoading || isRefreshing;
  const blockingError =
    view.kind === "error"
      ? view.error
      : view.kind === "success"
        ? view.refreshError
        : undefined;
  const snapshotMismatch =
    blockingError?.code === "AI_ANALYSIS_SNAPSHOT_MISMATCH";
  const disabled = isBusy || blockingError?.retryable === false;
  const title = isBusy
    ? isRefreshing
      ? "正在更新報告"
      : "正在產生報告"
    : snapshotMismatch
      ? "請先重新查詢公司資料"
      : blockingError
      ? blockingError.retryable
        ? "可重新嘗試"
        : "暫時無法產生"
      : view.kind === "success"
        ? "需要更新時再產生"
        : "需要時再產生";
  const helper = isBusy
    ? "請保留此頁；公司資料與 BizScore 仍可繼續查看。"
    : snapshotMismatch
      ? "AI 回應使用的資料版本與目前畫面不同。請回到上方重新查詢這家公司，確認最新資料後再產生分析。"
      : blockingError?.retryable === false
      ? "目前設定或回應未通過檢查，請聯絡系統管理者；既有公司資料不受影響。"
      : view.kind === "success"
        ? "重新產生會再次送出一筆 AI 請求；新報告成功前會保留目前版本。"
        : "每次點擊會送出一筆 AI 分析請求，不會在頁面載入時自動呼叫。";
  const buttonLabel = isInitialLoading
    ? "正在產生分析"
    : isRefreshing
      ? "正在重新產生"
      : snapshotMismatch
        ? "請先重新查詢公司"
        : blockingError?.retryable
        ? "重新嘗試"
        : view.kind === "success"
          ? "重新產生分析"
          : view.kind === "error"
            ? "目前無法產生"
            : "產生 AI 分析";

  return (
    <div className="ai-action-shell">
      {isRefreshing && (
        <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">
          正在重新產生 AI 分析，上一份報告會保留至更新完成。
        </p>
      )}
      <div className="ai-generate-row">
        <div>
          <strong>{title}</strong>
          <small id={descriptionId}>{helper}</small>
        </div>
        <button
          className={view.kind === "idle" ? "ai-primary-button" : "ai-secondary-button"}
          type="button"
          onClick={() => {
            if (!disabled) onGenerate();
          }}
          aria-describedby={descriptionId}
          aria-disabled={disabled}
          disabled={disabled}
        >
          {isBusy ? (
            <LoaderCircle className="spin" aria-hidden="true" size={18} />
          ) : view.kind === "success" || blockingError ? (
            <RefreshCw aria-hidden="true" size={17} />
          ) : (
            <Sparkles aria-hidden="true" size={18} />
          )}
          {buttonLabel}
        </button>
      </div>
    </div>
  );
}

function AnalysisReport({
  response,
}: {
  response: CompanyAnalysisResponse;
}) {
  const { input, analysis } = response.data;
  const isInsufficient = analysis.status === "insufficient_data";
  return (
    <div className="ai-report">
      <header className="ai-report-summary">
        <div>
          <span className={`ai-report-status${isInsufficient ? " insufficient" : ""}`}>
            {isInsufficient ? (
              <AlertTriangle aria-hidden="true" size={15} />
            ) : (
              <CheckCircle2 aria-hidden="true" size={15} />
            )}
            {isInsufficient
              ? "資料不足，提供查核方向"
              : "已通過格式、引用與禁用措辭檢查"}
          </span>
          {(input.source_meta.partial ||
            input.source_meta.warnings.length > 0 ||
            input.bizscore.provisional) && (
            <div className="ai-report-flags" aria-label="本次資料狀態提醒">
              {(input.source_meta.partial || input.source_meta.warnings.length > 0) && (
                <span>
                  <AlertTriangle aria-hidden="true" size={14} />
                  {input.source_meta.partial ? "來源含部分資料" : "來源含資料提醒"}
                </span>
              )}
              {input.bizscore.provisional && (
                <span>
                  <AlertTriangle aria-hidden="true" size={14} />
                  BizScore 為暫定結果
                </span>
              )}
            </div>
          )}
          <h4>{analysis.headline}</h4>
          <p>{analysis.overall_observation}</p>
        </div>
        <dl className="ai-report-snapshot">
          <div>
            <dt>分析時間</dt>
            <dd>{formatDateTime(response.meta.generated_at)}</dd>
          </div>
          <div>
            <dt>資料基準日</dt>
            <dd>{formatDate(input.bizscore.as_of)}</dd>
          </div>
        </dl>
      </header>

      <div className="ai-report-layout">
        <section className="ai-findings-section" aria-labelledby="ai-findings-title">
          <div className="ai-report-section-heading">
            <div>
              <p className="section-kicker">可回溯資料觀察</p>
              <h4 id="ai-findings-title">觀察重點</h4>
            </div>
            <span>{analysis.findings.length} 項</span>
          </div>
          {analysis.findings.length === 0 ? (
            <div className="ai-empty-findings">
              <Info aria-hidden="true" size={19} />
              <p>目前資料不足以形成可靠觀察，請直接查看合作前查核問題。</p>
            </div>
          ) : (
            <ol className="ai-findings-list">
              {analysis.findings.map((finding) => (
                <li key={finding.topic}>
                  <span className="ai-topic-label">{TOPIC_LABELS[finding.topic]}</span>
                  <h5>{finding.title}</h5>
                  <p>{finding.observation}</p>
                  {finding.caveat && (
                    <div className="ai-caveat">
                      <Info aria-hidden="true" size={15} />
                      <span>{finding.caveat}</span>
                    </div>
                  )}
                  <EvidenceDisclosure
                    paths={finding.evidence_paths}
                    input={input}
                  />
                </li>
              ))}
            </ol>
          )}
        </section>

        <section
          className="ai-verification-section"
          aria-labelledby="ai-verification-title"
        >
          <div className="ai-report-section-heading">
            <div>
              <p className="section-kicker">下一步可執行</p>
              <h4 id="ai-verification-title">合作前查核問題</h4>
            </div>
            <span>{analysis.verification_items.length} 題</span>
          </div>
          <ol className="ai-verification-list">
            {analysis.verification_items.map((item, index) => (
              <li key={`${item.question}-${index}`}>
                <div className="ai-question-topline">
                  <span className={`ai-priority ${priorityClass(item.priority)}`}>
                    {item.priority}
                  </span>
                  <span aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>
                </div>
                <strong>{item.question}</strong>
                <p>{item.reason}</p>
                <EvidenceDisclosure
                  paths={item.related_evidence_paths}
                  input={input}
                  compact
                />
              </li>
            ))}
          </ol>
        </section>
      </div>

      <section className="ai-limitations-section" aria-labelledby="ai-limitations-title">
        <div className="ai-report-section-heading">
          <div>
            <p className="section-kicker">判讀前請先確認</p>
            <h4 id="ai-limitations-title">資料限制與免責</h4>
          </div>
        </div>
        <ul className="ai-limitations-list">
          {analysis.limitations.map((limitation) => (
            <li key={limitation.code}>
              <ShieldAlert aria-hidden="true" size={17} />
              <span>
                <strong>{LIMITATION_LABELS[limitation.code]}</strong>
                <small>{limitation.message}</small>
                <EvidenceDisclosure
                  paths={limitation.related_evidence_paths}
                  input={input}
                  compact
                />
              </span>
            </li>
          ))}
        </ul>
        <div className="ai-fixed-disclaimer">
          <ShieldAlert aria-hidden="true" size={20} />
          <p>
            <strong>AI 內容使用聲明</strong>
            {analysis.disclaimer}
          </p>
        </div>
      </section>

      <footer className="ai-report-footer">
        <details className="ai-audit-details">
          <summary>
            <Database aria-hidden="true" size={16} />
            查看報告來源與版本
          </summary>
          <dl>
            <div>
              <dt>AI 服務</dt>
              <dd>{response.meta.provider}</dd>
            </div>
            <div>
              <dt>模型</dt>
              <dd>{response.meta.model}</dd>
            </div>
            <div>
              <dt>Prompt 版本</dt>
              <dd>v{response.meta.prompt_version}</dd>
            </div>
            <div>
              <dt>輸入／輸出契約</dt>
              <dd>
                v{response.meta.input_schema_version}／v
                {response.meta.output_schema_version}
              </dd>
            </div>
            <div>
              <dt>Benchmark</dt>
              <dd>{analysis.provenance.benchmark_catalog_version}</dd>
            </div>
            {response.meta.usage && (
              <div>
                <dt>本次 Token</dt>
                <dd>{formatInteger(response.meta.usage.total_tokens)}</dd>
              </div>
            )}
          </dl>
          <p>輸出格式、資料引用路徑與版本值均已由伺服器檢查；原始稽核欄位保留於 API。</p>
        </details>
      </footer>
    </div>
  );
}

function EvidenceDisclosure({
  paths,
  input,
  compact = false,
}: {
  paths: string[];
  input: CompanyAnalysisLLMInput;
  compact?: boolean;
}) {
  const rows = evidenceRows(paths, input);
  if (rows.length === 0) return null;
  return (
    <details className={`ai-evidence${compact ? " compact" : ""}`}>
      <summary>
        <FileCheck2 aria-hidden="true" size={15} />
        查看 {rows.length} 項資料依據
      </summary>
      <ul>
        {rows.map((row) => (
          <li key={row.path}>
            <span>{row.label}</span>
            <strong>{row.value}</strong>
          </li>
        ))}
      </ul>
    </details>
  );
}

function evidenceRows(
  paths: string[],
  input: CompanyAnalysisLLMInput,
): EvidenceRow[] {
  const rows: EvidenceRow[] = [];
  const seen = new Set<string>();
  for (const path of paths) {
    if (seen.has(path)) continue;
    const value = resolveJsonPointer(input, path);
    if (value === undefined) continue;
    const row = {
      path,
      label: evidenceLabel(path, input),
      value: formatEvidenceValue(path, value, input),
    };
    seen.add(path);
    rows.push(row);
  }
  return rows;
}

function resolveJsonPointer(root: unknown, pointer: string): unknown {
  if (!pointer.startsWith("/")) return undefined;
  let current = root;
  for (const rawSegment of pointer.slice(1).split("/")) {
    const segment = rawSegment.replaceAll("~1", "/").replaceAll("~0", "~");
    if (Array.isArray(current)) {
      if (!/^\d+$/.test(segment)) return undefined;
      current = current[Number(segment)];
    } else if (isRecord(current) && segment in current) {
      current = current[segment];
    } else {
      return undefined;
    }
  }
  return current;
}

function evidenceLabel(path: string, input: CompanyAnalysisLLMInput): string {
  const direct = EVIDENCE_LABELS[path];
  if (direct) return direct;

  const dimensionMatch = path.match(/^\/bizscore\/dimensions\/(\d+)\/(.+)$/);
  if (dimensionMatch) {
    const dimension = input.bizscore.dimensions[Number(dimensionMatch[1])];
    return dimension ? `${dimension.label}資料` : "BizScore 構面資料";
  }
  if (path.startsWith("/bizscore/peer_benchmark/dimension/")) {
    return "同業相對位置計分資料";
  }
  if (path.startsWith("/company/business_items/")) return "登記營業項目";
  if (path.startsWith("/source_meta/warnings/")) return "來源資料提醒";
  if (path.startsWith("/bizscore/warnings/")) return "BizScore 資料提醒";
  return path.startsWith("/company")
    ? "公司登記資料"
    : path.startsWith("/bizscore")
      ? "BizScore 資料"
      : "資料來源資訊";
}

function formatEvidenceValue(
  path: string,
  value: unknown,
  input: CompanyAnalysisLLMInput,
): string {
  if (value === null) return "資料未提供";
  if (path.startsWith("/source_meta/warnings/")) {
    return "來源資料含欄位缺漏或差異提醒";
  }
  if (path.startsWith("/bizscore/warnings/")) {
    return "BizScore 含資料限制提醒";
  }
  if (
    path.startsWith("/bizscore/missing_dimensions/") &&
    typeof value === "string"
  ) {
    const label = TOPIC_LABELS[value as AnalysisFindingTopic] ?? "部分";
    return `${label}構面資料不足`;
  }
  if (path === "/source_meta/partial" && typeof value === "boolean") {
    return value ? "是，部分欄位可能缺漏" : "否，來源資料完整";
  }
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "number") {
    if (path.includes("capital") && !path.includes("percentile")) {
      return new Intl.NumberFormat("zh-TW", {
        style: "currency",
        currency: "TWD",
        maximumFractionDigits: 0,
      }).format(value);
    }
    if (path.endsWith("coverage")) return `${Math.round(value * 100)}%`;
    if (path.includes("percentile") || path.endsWith("peer_index")) {
      return `PR ${formatDecimal(value)}`;
    }
    if (path.endsWith("sample_size")) return `${formatInteger(value)} 家`;
    if (path.endsWith("score")) return `${formatDecimal(value)} 分`;
    if (path.endsWith("company_age_years")) {
      return formatCompanyAge(
        input.company.established_at,
        input.source_meta.fetched_at,
        value,
      );
    }
    return formatDecimal(value);
  }
  if (typeof value === "string") {
    if (path === "/source_meta/data_freshness") {
      if (value === "stale_cache") return "即時資料暫不可用，使用最近成功快照";
      if (value === "fresh_cache") return "使用短期快取快照";
      if (value === "live") return "GCIS 即時取得";
    }
    if (path === "/source_meta/fallback_reason") {
      return "GCIS 即時更新暫時失敗";
    }
    if (/^\d{4}-\d{2}-\d{2}(?:T|$)/.test(value)) {
      return value.includes("T") ? formatDateTime(value) : formatDate(value);
    }
    if (path.includes("/evidence/")) return friendlyRawEvidence(value, input);
    return value || "資料未提供";
  }
  return "此資料已由伺服器核對";
}

function friendlyRawEvidence(
  value: string,
  input: CompanyAnalysisLLMInput,
): string {
  const separator = value.indexOf("=");
  if (separator <= 0) return "此資料已由伺服器核對";
  const key = value.slice(0, separator);
  const rawValue = value.slice(separator + 1);
  if (key === "rule") {
    return friendlyBizScoreRule(rawValue) ?? "符合既定的 BizScore 計分區間";
  }
  if (key === "company_age_years") {
    const years = Number(rawValue);
    return Number.isFinite(years)
      ? `成立年資：${formatCompanyAge(
          input.company.established_at,
          input.bizscore.as_of,
          years,
        )}`
      : "成立年資：資料未提供";
  }
  if (key === "capital.registered") {
    const capital = Number(rawValue);
    return Number.isFinite(capital)
      ? `登記資本額：${new Intl.NumberFormat("zh-TW", {
          style: "currency",
          currency: "TWD",
          maximumFractionDigits: 0,
        }).format(capital)}`
      : "登記資本額：資料未提供";
  }
  if (["last_changed_at", "established_at", "as_of"].includes(key)) {
    return `${
      key === "last_changed_at"
        ? "最後登記異動日"
        : key === "established_at"
          ? "核准設立日期"
          : "資料基準日"
    }：${formatDate(rawValue)}`;
  }
  if (key === "days_since_last_change") {
    const days = Number(rawValue);
    return Number.isFinite(days)
      ? `異動距基準日：${formatInteger(days)} 天`
      : "異動距基準日：資料未提供";
  }
  if (key === "peer.sample_size") {
    const sampleSize = Number(rawValue);
    return Number.isFinite(sampleSize)
      ? `有效同業樣本：${formatInteger(sampleSize)} 家`
      : "有效同業樣本：資料未提供";
  }
  if (["age_pr", "capital_pr", "peer_index"].includes(key)) {
    const percentile = Number(rawValue);
    return Number.isFinite(percentile)
      ? `同業相對位置：PR ${formatDecimal(percentile)}`
      : "同業相對位置：資料未提供";
  }
  const labels: Record<string, string> = {
    "status.code": "登記狀態代碼",
    "status.description": "登記狀態",
    official_status: "官方登記狀態",
    "peer.company_age_median": "同業成立年資中位數",
    "peer.registered_capital_median": "同業登記資本額中位數",
    industry_code: "比較產業代碼",
    "benchmark.version": "同業基準版本",
  };
  const label = labels[key];
  return label ? `${label}：${rawValue}` : "此資料已由伺服器核對";
}

function analysisErrorPresentation(error: ApiError): {
  title: string;
  helper: string;
} {
  if (error.code === "AI_ANALYSIS_SNAPSHOT_MISMATCH") {
    return {
      title: "公司資料版本與分析不一致",
      helper: "請回到上方重新查詢這家公司，確認最新公司資料與 BizScore 後再產生分析。",
    };
  }
  if (
    [
      "AI_ANALYSIS_NOT_CONFIGURED",
      "AI_ANALYSIS_CONFIGURATION_ERROR",
      "AI_PROVIDER_AUTH_FAILED",
    ].includes(error.code)
  ) {
    return {
      title: "AI 分析功能目前尚未啟用",
      helper: "上方 BizScore、同業比較與公司登記資料仍可正常使用。",
    };
  }
  if (
    [
      "AI_PROVIDER_REQUEST_REJECTED",
      "AI_ANALYSIS_INCOMPLETE",
      "AI_ANALYSIS_INVALID_OUTPUT",
      "AI_ANALYSIS_INVALID_RESPONSE",
      "AI_ANALYSIS_COMPANY_MISMATCH",
      "AI_PROVIDER_INVALID_RESPONSE",
    ].includes(error.code)
  ) {
    return {
      title: "本次沒有產生通過驗證的報告",
      helper: "系統已停止顯示不完整或未通過安全檢查的 AI 內容；原始資料不受影響。",
    };
  }
  if (error.code === "AI_PROVIDER_RATE_LIMITED") {
    return {
      title: "AI 分析服務目前繁忙",
      helper: "稍後可以重新嘗試；上方公司資料與分數會繼續保留。",
    };
  }
  return {
    title: error.status === 504 ? "AI 分析等待時間過久" : "目前無法完成 AI 分析",
    helper: "這不會影響已取得的公司資料與 BizScore；你可以稍後重新嘗試。",
  };
}

function priorityClass(priority: VerificationPriority): string {
  if (priority === "優先") return "priority-high";
  if (priority === "一般") return "priority-medium";
  return "priority-low";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat("zh-TW", { maximumFractionDigits: 0 }).format(value);
}

function formatDecimal(value: number): string {
  return new Intl.NumberFormat("zh-TW", { maximumFractionDigits: 1 }).format(value);
}

function formatDate(value: string): string {
  return value.replaceAll("-", "/");
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-TW", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "Asia/Taipei",
  }).format(date);
}
