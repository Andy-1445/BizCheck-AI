import {
  type Ref,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
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

import { ApiError, createCompanyComparisonAnalysis } from "./api";
import { formatCompanyAge } from "./formatters";
import type {
  CompanyComparisonAnalysisCacheMeta,
  CompanyComparisonAnalysisResponse,
  CompanyComparisonEvidencePath,
  CompanyComparisonLimitationCode,
  CompanyComparisonResponse,
  VerificationPriority,
} from "./types";

interface AIComparisonPanelProps {
  comparison: CompanyComparisonResponse;
}

type AIComparisonView =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; error: ApiError }
  | {
      kind: "success";
      response: CompanyComparisonAnalysisResponse;
      refreshing: boolean;
      refreshError?: ApiError;
    };

interface EvidenceRow {
  path: CompanyComparisonEvidencePath;
  label: string;
  value: string;
}

const FINDING_TOPIC_LABELS: Record<string, string> = {
  registration_status: "登記狀態",
  company_age: "成立時間",
  registered_capital_scale: "登記資本",
  registration_change_recency: "登記異動",
  peer_relative_position: "同業相對位置",
  data_completeness: "資料完整度",
  bizscore_context: "分數脈絡",
  peer_scope: "同業比較範圍",
};

const LIMITATION_LABELS: Record<CompanyComparisonLimitationCode, string> = {
  public_data_only: "僅限公開資料",
  ai_generated: "AI 產生內容",
  not_ranked: "不計算排名",
  partial_source_data: "來源含部分資料",
  provisional_score: "含暫定分數",
  no_numeric_score: "含無數字總分",
  cross_industry_comparison: "跨產業比較限制",
  benchmark_unavailable: "同業基準不可用",
};

const EVIDENCE_FIELD_LABELS: Record<string, string> = {
  "company.name": "公司名稱",
  "company.tax_id": "統一編號",
  "company.status.code": "登記狀態代碼",
  "company.status.description": "登記狀態",
  "company.established_at": "核准設立日期",
  "company.company_age_years": "成立年資",
  "company.capital.registered": "登記資本額",
  "company.last_changed_at": "最後登記異動日",
  "metrics.bizscore": "BizScore",
  "metrics.company_age_years": "成立年資",
  "metrics.bizscore_coverage": "可計分資料覆蓋率",
  "metrics.bizscore_provisional": "是否為暫定分數",
  "metrics.peer_index": "同業綜合相對位置",
  "metrics.industry_code": "主要產業代碼",
  "bizscore.band": "公開資料觀察區間",
  "bizscore.benchmark.sample_count": "同業樣本數",
  "bizscore.peer_benchmark.peer_index": "同業綜合相對位置",
  "source_meta.partial": "是否為部分資料",
  "source_meta.fetched_at": "公司資料快照時間",
  "source_meta.data_freshness": "資料即時狀態",
  "source_meta.fallback_reason": "即時資料不可用原因",
  "context.peer_comparison_scope": "同業比較範圍",
  "context.same_primary_industry": "是否為相同主要產業",
  "context.benchmark_as_of": "比較資料基準日",
  "context.benchmark_catalog_version": "Benchmark 版本",
  "meta.has_partial_source_data": "是否含部分來源資料",
  "meta.has_provisional_scores": "是否含暫定分數",
  "meta.has_unscored_companies": "是否含無數字總分",
};

export default function AIComparisonPanel({ comparison }: AIComparisonPanelProps) {
  const [view, setView] = useState<AIComparisonView>({ kind: "idle" });
  const controllerRef = useRef<AbortController | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const errorRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const actionHelpId = useId();
  const taxIds = useMemo(
    () => comparison.data.items.map((item) => item.company.tax_id),
    [comparison],
  );
  useEffect(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setView({ kind: "idle" });
  }, [comparison]);

  useEffect(() => () => controllerRef.current?.abort(), []);

  useEffect(() => {
    if (
      view.kind === "error" ||
      (view.kind === "success" && !view.refreshing && view.refreshError)
    ) {
      errorRef.current?.focus();
    } else if (view.kind === "success" && !view.refreshing) {
      resultRef.current?.focus();
    }
  }, [view]);

  const generate = async (forceRefresh: boolean) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const preserved = view.kind === "success" ? view.response : null;
    setView(
      preserved
        ? { kind: "success", response: preserved, refreshing: true }
        : { kind: "loading" },
    );
    try {
      const response = await createCompanyComparisonAnalysis(
        comparison,
        forceRefresh,
        controller.signal,
      );
      if (controller.signal.aborted) return;
      setView({ kind: "success", response, refreshing: false });
    } catch (error) {
      if (controller.signal.aborted) return;
      const publicError = toApiError(error);
      setView(
        preserved
          ? {
              kind: "success",
              response: preserved,
              refreshing: false,
              refreshError: publicError,
            }
          : { kind: "error", error: publicError },
      );
    }
  };

  const busy = view.kind === "loading" || (view.kind === "success" && view.refreshing);

  return (
    <section
      className={`ai-comparison-panel ${view.kind}`}
      aria-labelledby={titleId}
      aria-busy={busy}
    >
      <header className="ai-comparison-heading">
        <div className="ai-comparison-title-lockup">
          <span className="ai-comparison-mark" aria-hidden="true">
            <Sparkles size={20} />
          </span>
          <div>
            <p className="section-kicker">比較表的 AI 邊註</p>
            <h3 id={titleId}>AI 查核備忘錄</h3>
          </div>
        </div>
        <span className="ai-comparison-role">附註，不是裁判</span>
      </header>

      <div className="ai-comparison-boundary">
        <ShieldCheck aria-hidden="true" size={19} />
        <p>
          <strong>上方比較表永遠是原始依據。</strong>
          AI 只把既有登記資料、BizScore 與同業基準整理成觀察和查證問題；不重算分數、不排勝負，也不替你選擇合作對象。
        </p>
      </div>

      {view.kind === "idle" && (
        <AIComparisonIntro companyCount={taxIds.length} />
      )}
      {view.kind === "loading" && <AIComparisonLoading />}
      {view.kind === "error" && (
        <AIComparisonError error={view.error} focusRef={errorRef} />
      )}
      {view.kind === "success" && (
        <div ref={resultRef} tabIndex={-1} className="ai-comparison-result">
          {view.response.data.analysis ? (
            <AIComparisonReport response={view.response} />
          ) : (
            <AIComparisonFallback response={view.response} />
          )}
          {view.refreshError && (
            <AIComparisonError
              error={view.refreshError}
              focusRef={errorRef}
              preserved
            />
          )}
        </div>
      )}

      <AIComparisonAction
        view={view}
        onGenerate={generate}
        descriptionId={actionHelpId}
      />
    </section>
  );
}

function AIComparisonIntro({ companyCount }: { companyCount: number }) {
  return (
    <div className="ai-comparison-intro">
      <p>
        需要時再產生一份 {companyCount} 家公司的查核備忘錄。頁面載入與公司比較完成時都不會自動呼叫 AI。
      </p>
      <div className="ai-comparison-preview" aria-label="備忘錄包含內容">
        <div>
          <FileCheck2 aria-hidden="true" size={19} />
          <span><strong>逐家公司觀察</strong><small>每一點都能展開已核對的資料依據</small></span>
        </div>
        <div>
          <Database aria-hidden="true" size={19} />
          <span><strong>共同與差異脈絡</strong><small>只說明資料差異，不把公司排成名次</small></span>
        </div>
        <div>
          <ListChecks aria-hidden="true" size={19} />
          <span><strong>合作前查證問題</strong><small>把公開資料無法回答的部分列成待辦</small></span>
        </div>
      </div>
    </div>
  );
}

function AIComparisonLoading() {
  return (
    <div className="ai-comparison-loading" role="status" aria-live="polite">
      <LoaderCircle className="spin" aria-hidden="true" size={28} />
      <div>
        <h4>正在整理 AI 查核備忘錄</h4>
        <p>上方比較表仍可查看；系統正在核對引用並檢查禁用措辭。</p>
      </div>
    </div>
  );
}

function AIComparisonError({
  error,
  focusRef,
  preserved = false,
}: {
  error: ApiError;
  focusRef?: Ref<HTMLDivElement>;
  preserved?: boolean;
}) {
  return (
    <div
      ref={focusRef}
      className={`ai-comparison-error${preserved ? " preserved" : ""}`}
      role="alert"
      tabIndex={focusRef ? -1 : undefined}
    >
      <AlertTriangle aria-hidden="true" size={21} />
      <div>
        <p className="ai-comparison-error-code">
          {preserved ? "更新未完成，已保留原備忘錄" : "AI 備忘錄未完成"} · {error.code}
        </p>
        <h4>AI 內容目前無法顯示</h4>
        <p>{error.message}</p>
        <small>原本的公司 PK 比較表沒有被移除或改寫，仍可直接作為查核依據。</small>
      </div>
    </div>
  );
}

function AIComparisonFallback({
  response,
}: {
  response: CompanyComparisonAnalysisResponse;
}) {
  const fallback = response.data.fallback;
  if (!fallback) return null;
  const wasNotCalled = fallback.code === "not_configured";
  return (
    <div className="ai-comparison-fallback" role="status">
      <div className="ai-comparison-fallback-label">
        <ShieldAlert aria-hidden="true" size={17} />
        非 AI 結果 · 確定性備援
      </div>
      <h4>{fallback.title}</h4>
      <p>{fallback.message}</p>
      <div className="ai-comparison-fallback-note">
        <Info aria-hidden="true" size={17} />
        <p>
          {wasNotCalled
            ? "本次未呼叫 AI 模型；系統只保留上方已驗證的比較表。"
            : "本次未顯示任何模型內容；系統已退回上方已驗證的比較表。"}
          這不是 AI 建議，也沒有替公司排名或選擇合作對象。
        </p>
      </div>
    </div>
  );
}

function AIComparisonReport({
  response,
}: {
  response: CompanyComparisonAnalysisResponse;
}) {
  const analysis = response.data.analysis;
  if (!analysis) return null;
  const comparison = response.data.comparison;
  const isInsufficient = analysis.status === "insufficient_data";
  const groupedCompanyObservations = comparison.data.items.map((item, index) => ({
    item,
    letter: String.fromCharCode(65 + index),
    observations: analysis.company_observations.filter(
      (observation) => observation.tax_id === item.company.tax_id,
    ),
  }));
  return (
    <div className="ai-comparison-report">
      <header className="ai-comparison-summary">
        <div>
          <div className="ai-comparison-status-row">
            <span className={`ai-comparison-output-status${isInsufficient ? " insufficient" : ""}`}>
              {isInsufficient ? (
                <AlertTriangle aria-hidden="true" size={15} />
              ) : (
                <CheckCircle2 aria-hidden="true" size={15} />
              )}
              {isInsufficient ? "資料不足，改列查證方向" : "已通過引用與安全檢查"}
            </span>
            <CacheBadge cache={response.meta.cache} />
          </div>
          <h4>{analysis.headline}</h4>
          <p>{analysis.overall_observation}</p>
        </div>
        <dl className="ai-comparison-snapshot">
          <div><dt>內容產生時間</dt><dd>{formatDateTime(response.meta.generated_at)}</dd></div>
          <div><dt>資料基準日</dt><dd>{formatDate(analysis.provenance.data_as_of)}</dd></div>
        </dl>
      </header>

      {response.meta.cache.status === "stale" && (
        <div className="ai-comparison-stale-warning" role="status">
          <AlertTriangle aria-hidden="true" size={19} />
          <p>
            <strong>目前顯示已驗證的舊快取。</strong>
            即時 AI 服務未完成，因此保留先前通過安全檢查的內容；產生時間仍顯示原始時間，請和上方最新比較表一起核對。
          </p>
        </div>
      )}

      <section className="ai-comparison-section" aria-labelledby="ai-company-notes-title">
        <div className="ai-comparison-section-heading">
          <div><p className="section-kicker">按 A／B／C 分頁的邊註</p><h4 id="ai-company-notes-title">逐家公司觀察</h4></div>
          <span>{analysis.company_observations.length} 項</span>
        </div>
        <div className="ai-company-note-grid">
          {groupedCompanyObservations.map(({ item, letter, observations }) => (
            <article key={item.company.tax_id}>
              <header>
                <span className="comparison-letter" aria-hidden="true">{letter}</span>
                <div><h5>{item.company.name}</h5><small>統編 {item.company.tax_id}</small></div>
              </header>
              {observations.length === 0 ? (
                <p className="ai-comparison-empty-note">目前沒有足夠資料形成可核對的公司觀察。</p>
              ) : (
                <ol>
                  {observations.map((observation) => (
                    <li key={`${observation.topic}-${observation.title}`}>
                      <span>{FINDING_TOPIC_LABELS[observation.topic]}</span>
                      <h6>{observation.title}</h6>
                      <p>{observation.observation}</p>
                      {observation.caveat && <small className="ai-comparison-caveat">{observation.caveat}</small>}
                      <ComparisonEvidence paths={observation.evidence_paths} comparison={comparison} />
                    </li>
                  ))}
                </ol>
              )}
            </article>
          ))}
        </div>
      </section>

      <div className="ai-comparison-two-column">
        <section className="ai-comparison-section" aria-labelledby="ai-cross-notes-title">
          <div className="ai-comparison-section-heading">
            <div><p className="section-kicker">並列脈絡，不是名次</p><h4 id="ai-cross-notes-title">共同點與差異</h4></div>
            <span>{analysis.comparison_observations.length} 項</span>
          </div>
          <ol className="ai-comparison-observation-list">
            {analysis.comparison_observations.map((observation) => (
              <li key={`${observation.topic}-${observation.title}`}>
                <span>{FINDING_TOPIC_LABELS[observation.topic]}</span>
                <h5>{observation.title}</h5>
                <p>{observation.observation}</p>
                {observation.caveat && <small className="ai-comparison-caveat">{observation.caveat}</small>}
                <ComparisonEvidence paths={observation.evidence_paths} comparison={comparison} />
              </li>
            ))}
          </ol>
        </section>

        <section className="ai-comparison-section ai-comparison-questions" aria-labelledby="ai-pk-questions-title">
          <div className="ai-comparison-section-heading">
            <div><p className="section-kicker">公開資料無法代答</p><h4 id="ai-pk-questions-title">合作前查證問題</h4></div>
            <span>{analysis.verification_items.length} 題</span>
          </div>
          <ol>
            {analysis.verification_items.map((item, index) => (
              <li key={`${item.question}-${index}`}>
                <div><span className={`ai-priority ${priorityClass(item.priority)}`}>{item.priority}</span><small>{String(index + 1).padStart(2, "0")}</small></div>
                <h5>{item.question}</h5>
                <p>{item.reason}</p>
                <ComparisonEvidence paths={item.related_evidence_paths} comparison={comparison} />
              </li>
            ))}
          </ol>
        </section>
      </div>

      <section className="ai-comparison-limitations" aria-labelledby="ai-pk-limitations-title">
        <div className="ai-comparison-section-heading">
          <div><p className="section-kicker">使用前先確認</p><h4 id="ai-pk-limitations-title">限制與來源</h4></div>
        </div>
        <ul>
          {analysis.limitations.map((limitation) => (
            <li key={limitation.code}>
              <ShieldAlert aria-hidden="true" size={17} />
              <span><strong>{LIMITATION_LABELS[limitation.code]}</strong><small>{limitation.message}</small></span>
            </li>
          ))}
        </ul>
        <div className="ai-comparison-disclaimer">
          <ShieldAlert aria-hidden="true" size={20} />
          <p><strong>AI 比較內容使用聲明</strong>{analysis.disclaimer}</p>
        </div>
      </section>

      <footer className="ai-comparison-audit">
        <details>
          <summary><Database aria-hidden="true" size={16} />查看快取、模型與版本</summary>
          <dl>
            <div><dt>AI 服務</dt><dd>{response.meta.provider}</dd></div>
            <div><dt>模型</dt><dd>{response.meta.model}</dd></div>
            <div><dt>Prompt</dt><dd>v{response.meta.prompt_version}</dd></div>
            <div><dt>快取狀態</dt><dd>{cacheLabel(response.meta.cache.status)}</dd></div>
            <div><dt>原始產生時間</dt><dd>{formatDateTime(response.meta.generated_at)}</dd></div>
            <div><dt>Benchmark</dt><dd>{analysis.provenance.benchmark_catalog_version}</dd></div>
          </dl>
          <p>內容只引用此份比較回應中的結構化欄位；完整雜湊與快取鍵保留於 API 稽核欄位。</p>
        </details>
      </footer>
    </div>
  );
}

function CacheBadge({ cache }: { cache: CompanyComparisonAnalysisCacheMeta }) {
  const tone = cache.status === "stale" ? " stale" : cache.status === "hit" ? " fresh" : "";
  return (
    <span className={`ai-comparison-cache-badge${tone}`}>
      <Database aria-hidden="true" size={14} />
      {cacheLabel(cache.status)}
    </span>
  );
}

function AIComparisonAction({
  view,
  onGenerate,
  descriptionId,
}: {
  view: AIComparisonView;
  onGenerate: (forceRefresh: boolean) => Promise<void>;
  descriptionId: string;
}) {
  const busy = view.kind === "loading" || (view.kind === "success" && view.refreshing);
  const response = view.kind === "success" ? view.response : null;
  const fallback = response?.data.fallback ?? null;
  const error = view.kind === "error" ? view.error : view.kind === "success" ? view.refreshError : undefined;
  const retryable = fallback?.retryable ?? error?.retryable ?? true;
  const snapshotMismatch = error?.code === "AI_COMPARISON_SNAPSHOT_MISMATCH";
  const hasResult = Boolean(response);
  const forceRefresh = Boolean(response?.data.analysis);
  const disabled = busy || ((fallback || error) && !retryable);
  const label = busy
    ? response
      ? "正在更新備忘錄"
      : "正在產生備忘錄"
    : fallback || error
      ? retryable
        ? "重新嘗試 AI 備忘錄"
        : snapshotMismatch
          ? "請先重跑公司比較"
          : "AI 備忘錄目前未啟用"
      : response?.data.analysis
        ? "忽略快取，重新產生"
        : "產生 AI 查核備忘錄";
  const help = busy
    ? "上方確定性比較表仍可使用；新內容完成前會保留目前畫面。"
    : snapshotMismatch
      ? "公司資料已變動，請回到上方調整區重新按一次「開始比較」，再產生新的 AI 備忘錄。"
      : forceRefresh
      ? "重新請 AI 整理這份資料；更新成功前會保留原報告，避免閱讀中斷。"
      : "只有點擊此按鈕才會送出 AI 請求；公司比較與頁面載入不會自動呼叫。";
  return (
    <div className="ai-comparison-action">
      <div>
        <strong>{hasResult ? "需要時才更新" : "由你決定是否使用 AI"}</strong>
        <small id={descriptionId}>{help}</small>
      </div>
      <button
        type="button"
        className={hasResult ? "ai-secondary-button" : "ai-primary-button"}
        onClick={() => {
          if (!disabled) void onGenerate(forceRefresh || Boolean(fallback));
        }}
        aria-describedby={descriptionId}
        disabled={Boolean(disabled)}
      >
        {busy ? <LoaderCircle className="spin" aria-hidden="true" size={18} /> : hasResult || error ? <RefreshCw aria-hidden="true" size={17} /> : <Sparkles aria-hidden="true" size={18} />}
        {label}
      </button>
    </div>
  );
}

function ComparisonEvidence({
  paths,
  comparison,
}: {
  paths: CompanyComparisonEvidencePath[];
  comparison: CompanyComparisonResponse;
}) {
  const rows = evidenceRows(paths, comparison);
  if (rows.length === 0) return null;
  return (
    <details className="ai-comparison-evidence">
      <summary><FileCheck2 aria-hidden="true" size={15} />查看 {rows.length} 項資料依據</summary>
      <ul>
        {rows.map((row) => <li key={row.path}><span>{row.label}</span><strong>{row.value}</strong></li>)}
      </ul>
    </details>
  );
}

function evidenceRows(
  paths: CompanyComparisonEvidencePath[],
  comparison: CompanyComparisonResponse,
): EvidenceRow[] {
  const root = { comparison };
  const seen = new Set<string>();
  return paths.flatMap((path) => {
    if (seen.has(path)) return [];
    seen.add(path);
    const value = resolveJsonPointer(root, path);
    if (value === undefined) return [];
    return [{
      path,
      label: evidenceLabel(path, comparison),
      value: formatEvidenceValue(path, value, comparison),
    }];
  });
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

function evidenceLabel(path: string, comparison: CompanyComparisonResponse): string {
  const itemMatch = path.match(/^\/comparison\/data\/items\/(\d+)\/(.+)$/);
  if (itemMatch) {
    const index = Number(itemMatch[1]);
    const item = comparison.data.items[index];
    const field = itemMatch[2].replaceAll("/", ".");
    const label = EVIDENCE_FIELD_LABELS[field] ?? inferEvidenceLabel(field);
    return item ? `${String.fromCharCode(65 + index)}｜${item.company.name} · ${label}` : label;
  }
  const contextMatch = path.match(/^\/comparison\/data\/(context\/.+)$/);
  if (contextMatch) {
    const field = contextMatch[1].replaceAll("/", ".");
    return EVIDENCE_FIELD_LABELS[field] ?? inferEvidenceLabel(field);
  }
  const metaMatch = path.match(/^\/comparison\/(meta\/.+)$/);
  if (metaMatch) {
    const field = metaMatch[1].replaceAll("/", ".");
    return EVIDENCE_FIELD_LABELS[field] ?? inferEvidenceLabel(field);
  }
  return "已核對的比較資料";
}

function inferEvidenceLabel(field: string): string {
  if (field.includes("dimensions")) return "BizScore 構面資料";
  if (field.includes("warnings")) return "來源資料提醒";
  if (field.includes("business_items")) return "登記營業項目";
  if (field.includes("peer_benchmark")) return "同業基準資料";
  if (field.includes("benchmark")) return "Benchmark 資料";
  return "已核對的結構化欄位";
}

function formatEvidenceValue(
  path: string,
  value: unknown,
  comparison: CompanyComparisonResponse,
): string {
  if (value === null) return "資料未提供";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "number") {
    if (path.endsWith("company_age_years")) {
      const itemIndex = path.match(/^\/comparison\/data\/items\/(\d+)\//)?.[1];
      const item = itemIndex === undefined
        ? undefined
        : comparison.data.items[Number(itemIndex)];
      return formatCompanyAge(
        item?.company.established_at ?? null,
        item?.bizscore.as_of ?? comparison.data.context.benchmark_as_of,
        value,
      );
    }
    if (path.includes("capital") && !path.includes("percentile")) {
      return new Intl.NumberFormat("zh-TW", { style: "currency", currency: "TWD", maximumFractionDigits: 0 }).format(value);
    }
    if (path.endsWith("coverage")) return `${Math.round(value * 100)}%`;
    if (path.endsWith("peer_index") || path.includes("percentile")) return `PR ${formatNumber(value)}`;
    if (path.endsWith("sample_count") || path.endsWith("sample_size")) return `${formatInteger(value)} 家`;
    if (path.endsWith("score")) return `${formatNumber(value)} 分`;
    return formatNumber(value);
  }
  if (typeof value === "string") {
    if (path.endsWith("/source_meta/data_freshness")) {
      if (value === "stale_cache") return "即時資料暫不可用，使用最近成功快照";
      if (value === "fresh_cache") return "使用短期快取快照";
      if (value === "live") return "GCIS 即時取得";
    }
    if (path.endsWith("/source_meta/fallback_reason")) {
      return "GCIS 即時更新暫時失敗";
    }
    if (/^\d{4}-\d{2}-\d{2}(?:T|$)/.test(value)) return value.includes("T") ? formatDateTime(value) : formatDate(value);
    return value || "資料未提供";
  }
  if (Array.isArray(value)) return value.length === 0 ? "無" : `共 ${value.length} 項已核對資料`;
  return "已由伺服器核對的結構化資料";
}

function cacheLabel(status: CompanyComparisonAnalysisCacheMeta["status"]): string {
  if (status === "hit") return "有效快取";
  if (status === "stale") return "已驗證舊快取";
  if (status === "bypass") return "已略過快取，本次產生";
  return "本次產生";
}

function priorityClass(priority: VerificationPriority): string {
  if (priority === "優先") return "priority-high";
  if (priority === "一般") return "priority-medium";
  return "priority-low";
}

function toApiError(error: unknown): ApiError {
  return error instanceof ApiError
    ? error
    : new ApiError(0, {
        code: "AI_COMPARISON_UNEXPECTED_ERROR",
        message: "產生 AI 查核備忘錄時發生非預期錯誤，請稍後再試。",
        retryable: true,
      });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat("zh-TW", { maximumFractionDigits: 0 }).format(value);
}

function formatNumber(value: number): string {
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
