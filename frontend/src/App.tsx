import {
  type FormEvent,
  type MouseEvent,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  AlertTriangle,
  ArrowRight,
  Building2,
  CheckCircle2,
  Clock3,
  Database,
  ExternalLink,
  FileSearch,
  GitCompareArrows,
  Info,
  Landmark,
  ListChecks,
  LoaderCircle,
  MapPin,
  RotateCcw,
  Search,
  ShieldCheck,
  UserRound,
} from "lucide-react";

import { ApiError, getCompanyBizScore, searchCompanies } from "./api";
import AIAnalysisPanel from "./AIAnalysisPanel";
import BizScorePanel from "./BizScorePanel";
import CompanyComparisonPanel, {
  type ComparisonSeed,
} from "./CompanyComparisonPanel";
import { formatCompanyAge } from "./formatters";
import StaleDataNotice from "./StaleDataNotice";
import type {
  CompanyBizScoreResponse,
  CompanyData,
  CompanySearchItem,
  CompanySearchResponse,
  ResponseMeta,
} from "./types";

type ViewState =
  | { kind: "idle" }
  | { kind: "loading"; message: string }
  | { kind: "results"; response: CompanySearchResponse }
  | { kind: "company"; response: CompanyBizScoreResponse }
  | { kind: "error"; error: ApiError };

interface RetryAction {
  kind: "search" | "company";
  value: string;
}

type AppMode = "single" | "comparison";

const SOURCE_URL = "https://data.gcis.nat.gov.tw/";

export default function App() {
  const [mode, setMode] = useState<AppMode>("single");
  const [comparisonSeed, setComparisonSeed] = useState<ComparisonSeed | null>(
    null,
  );
  const [query, setQuery] = useState("");
  const [fieldError, setFieldError] = useState<string | null>(null);
  const [view, setView] = useState<ViewState>({ kind: "idle" });
  const [retryAction, setRetryAction] = useState<RetryAction | null>(null);
  const requestController = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const contentRef = useRef<HTMLElement>(null);
  const mainRef = useRef<HTMLElement>(null);
  const comparisonSeedNonce = useRef(0);

  useEffect(() => {
    return () => requestController.current?.abort();
  }, []);

  useEffect(() => {
    if (view.kind !== "idle" && view.kind !== "loading") {
      contentRef.current?.focus();
    }
  }, [view]);

  const submitQuery = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const normalized = query.trim();
    setFieldError(null);

    if (!normalized) {
      setFieldError("請輸入公司名稱或 8 碼統一編號。");
      inputRef.current?.focus();
      return;
    }

    if (/^\d+$/.test(normalized)) {
      if (!/^\d{8}$/.test(normalized)) {
        setFieldError("統一編號必須是 8 碼數字。");
        inputRef.current?.focus();
        return;
      }
      await loadCompany(normalized);
      return;
    }

    if (normalized.length < 2) {
      setFieldError("公司名稱至少需要 2 個字元。");
      inputRef.current?.focus();
      return;
    }
    await loadSearchResults(normalized);
  };

  const loadSearchResults = async (keyword: string) => {
    const controller = beginRequest();
    setView({ kind: "loading", message: "正在比對核准設立公司…" });
    setRetryAction({ kind: "search", value: keyword });

    try {
      const response = await searchCompanies(keyword, controller.signal);
      if (
        !controller.signal.aborted &&
        requestController.current === controller
      ) {
        setView({ kind: "results", response });
      }
    } catch (error) {
      handleRequestError(error, controller);
    }
  };

  const loadCompany = async (taxId: string) => {
    const controller = beginRequest();
    setView({ kind: "loading", message: "正在計算公開資料體質指標…" });
    setRetryAction({ kind: "company", value: taxId });

    try {
      const response = await getCompanyBizScore(taxId, controller.signal);
      if (
        !controller.signal.aborted &&
        requestController.current === controller
      ) {
        setQuery(taxId);
        setView({ kind: "company", response });
      }
    } catch (error) {
      handleRequestError(error, controller);
    }
  };

  const beginRequest = () => {
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    return controller;
  };

  const handleRequestError = (error: unknown, controller: AbortController) => {
    if (
      controller.signal.aborted ||
      requestController.current !== controller
    ) {
      return;
    }
    if (error instanceof ApiError) {
      setView({ kind: "error", error });
      return;
    }
    setView({
      kind: "error",
      error: new ApiError(0, {
        code: "UNEXPECTED_ERROR",
        message: "查詢時發生非預期錯誤，請重新再試一次。",
        retryable: true,
      }),
    });
  };

  const retry = () => {
    if (!retryAction) return;
    if (retryAction.kind === "company") {
      void loadCompany(retryAction.value);
    } else {
      void loadSearchResults(retryAction.value);
    }
  };

  const editQuery = () => {
    requestController.current?.abort();
    requestController.current = null;
    setRetryAction(null);
    setFieldError(null);
    setView({ kind: "idle" });
    window.setTimeout(() => {
      inputRef.current?.focus();
      inputRef.current?.scrollIntoView({
        behavior: preferredScrollBehavior(),
        block: "center",
      });
    }, 0);
  };

  const reset = () => {
    requestController.current?.abort();
    requestController.current = null;
    setMode("single");
    setComparisonSeed(null);
    setQuery("");
    setFieldError(null);
    setRetryAction(null);
    setView({ kind: "idle" });
    window.setTimeout(() => inputRef.current?.focus(), 0);
  };

  const switchMode = (nextMode: AppMode) => {
    if (nextMode === mode) return;
    requestController.current?.abort();
    requestController.current = null;
    if (view.kind === "loading") setView({ kind: "idle" });
    if (nextMode === "comparison" && view.kind === "company") {
      comparisonSeedNonce.current += 1;
      setComparisonSeed({
        taxId: view.response.data.company.tax_id,
        name: view.response.data.company.name,
        nonce: comparisonSeedNonce.current,
      });
    }
    setMode(nextMode);
  };

  const switchToComparison = (company: CompanyData) => {
    requestController.current?.abort();
    requestController.current = null;
    comparisonSeedNonce.current += 1;
    setComparisonSeed({
      taxId: company.tax_id,
      name: company.name,
      nonce: comparisonSeedNonce.current,
    });
    setMode("comparison");
  };

  const viewSingleCompany = (taxId: string) => {
    setMode("single");
    void loadCompany(taxId);
  };

  const railStep =
    view.kind === "company" ? 3 : view.kind === "results" ? 2 : 1;
  const isLoading = view.kind === "loading";

  const focusMainContent = (event: MouseEvent<HTMLAnchorElement>) => {
    event.preventDefault();
    const main = mainRef.current;
    if (!main) return;
    main.focus({ preventScroll: true });
    main.scrollIntoView({
      behavior: preferredScrollBehavior(),
      block: "start",
    });
  };

  return (
    <div className="app-shell">
      <a
        className="skip-link"
        href="#main-content"
        onClick={focusMainContent}
      >
        跳到主要內容
      </a>
      <header className="site-header">
        <div className="header-inner">
          <button className="brand" type="button" onClick={reset}>
            <span className="brand-mark" aria-hidden="true">
              <ShieldCheck size={22} strokeWidth={1.8} />
            </span>
            <span>
              <strong>BizCheck AI</strong>
              <small>企業公開資料查核</small>
            </span>
          </button>
          <div className="scope-badge">
            <Database aria-hidden="true" size={16} />
            資料來源：GCIS
          </div>
        </div>
      </header>

      <main
        ref={mainRef}
        className="main-content"
        id="main-content"
        tabIndex={-1}
      >
        <WorkspaceSwitcher mode={mode} onChange={switchMode} />

        {mode === "comparison" ? (
          <CompanyComparisonPanel
            seed={comparisonSeed}
            onViewCompany={viewSingleCompany}
          />
        ) : (
          <>
          <section className="search-stage" aria-labelledby="page-title">
          <div className="thesis">
            <h1 id="page-title" className="thesis-eyebrow">合作之前的第一道公開資料查核</h1>
            <p>
              輸入公司名稱或統一編號，查看登記資料、BizScore 分數拆解、同業相對位置與 AI 合作前查核報告。
            </p>
          </div>

          <form className="search-form" onSubmit={submitQuery} noValidate>
            <label htmlFor="company-query">公司名稱或統一編號</label>
            <div className="search-control">
              <Search className="search-icon" aria-hidden="true" size={21} />
              <input
                ref={inputRef}
                id="company-query"
                value={query}
                onChange={(event) => {
                  requestController.current?.abort();
                  requestController.current = null;
                  setQuery(event.target.value);
                  if (view.kind !== "idle") {
                    setRetryAction(null);
                    setView({ kind: "idle" });
                  }
                  if (fieldError) setFieldError(null);
                }}
                aria-describedby={
                  fieldError ? "query-help query-error" : "query-help"
                }
                aria-invalid={fieldError ? "true" : "false"}
                autoComplete="off"
                placeholder="例如：宏碁、20828393"
              />
              <button type="submit" disabled={isLoading}>
                {isLoading ? (
                  <LoaderCircle
                    className="spin"
                    aria-hidden="true"
                    size={19}
                  />
                ) : (
                  <Search aria-hidden="true" size={19} />
                )}
                {isLoading ? "查詢中" : "查詢公司"}
              </button>
            </div>
            <div className="form-support">
              <p id="query-help">
                名稱搜尋目前只列出狀態為「核准設立」的公司。
              </p>
              {fieldError && (
                <p id="query-error" className="field-error" role="alert">
                  {fieldError}
                </p>
              )}
            </div>
          </form>

          <VerificationRail currentStep={railStep} />
        </section>

        <section
          ref={contentRef}
          className="result-stage"
          tabIndex={-1}
          aria-busy={isLoading}
        >
          {view.kind === "idle" && <IdlePanel />}
          {view.kind === "loading" && <LoadingPanel message={view.message} />}
          {view.kind === "results" && (
            <SearchResults
              response={view.response}
              query={query}
              onSelect={(taxId) => void loadCompany(taxId)}
              onEdit={editQuery}
            />
          )}
          {view.kind === "company" && (
            <CompanyPanel
              response={view.response}
              onCompare={switchToComparison}
              onReset={reset}
            />
          )}
          {view.kind === "error" && (
            <ErrorPanel error={view.error} onRetry={retry} onEdit={editQuery} />
          )}
        </section>
          </>
        )}
      </main>

      <footer className="site-footer">
        <div>
          <ShieldCheck aria-hidden="true" size={18} />
          <p>
            BizCheck AI 整理政府公開登記資料，結果不代表信用評等、履約保證或倒閉預測。
          </p>
        </div>
        <a href={SOURCE_URL} target="_blank" rel="noreferrer">
          查看官方資料來源
          <ExternalLink aria-hidden="true" size={15} />
        </a>
      </footer>
    </div>
  );
}

function WorkspaceSwitcher({
  mode,
  onChange,
}: {
  mode: AppMode;
  onChange: (mode: AppMode) => void;
}) {
  return (
    <nav className="workspace-switcher" aria-label="選擇查核模式">
      <span>查核模式</span>
      <div>
        <button
          type="button"
          className={mode === "single" ? "active" : ""}
          aria-pressed={mode === "single"}
          onClick={() => onChange("single")}
        >
          <FileSearch aria-hidden="true" size={17} />
          單一公司查核
        </button>
        <button
          type="button"
          className={mode === "comparison" ? "active" : ""}
          aria-pressed={mode === "comparison"}
          onClick={() => onChange("comparison")}
        >
          <GitCompareArrows aria-hidden="true" size={17} />
          2–3 家公司比較
        </button>
      </div>
    </nav>
  );
}

function VerificationRail({ currentStep }: { currentStep: number }) {
  const steps = ["輸入查詢", "選擇公司", "檢視體質與 AI 分析"];
  return (
    <ol className="verification-rail" aria-label="公司查核進度">
      {steps.map((step, index) => {
        const stepNumber = index + 1;
        const isActive = stepNumber <= currentStep;
        return (
          <li className={isActive ? "active" : ""} key={step}>
            <span aria-hidden="true">{isActive ? <CheckCircle2 size={18} /> : stepNumber}</span>
            <strong>{step}</strong>
          </li>
        );
      })}
    </ol>
  );
}

function IdlePanel() {
  const facts = [
    [Building2, "登記身分", "名稱、統編與目前登記狀態"],
    [Landmark, "BizScore 拆解", "五個可回溯公開欄位的評分構面"],
    [ListChecks, "同業位置", "年資、資本額 PR 與同業中位數"],
    [Clock3, "AI 查核報告", "依已驗證資料整理觀察、問題與限制"],
  ] as const;

  return (
    <div className="idle-panel">
      <div className="section-heading">
        <p className="section-kicker">查詢結果會整理成</p>
        <h2>一張可核對的公司登記摘要</h2>
      </div>
      <div className="fact-preview-grid">
        {facts.map(([Icon, title, description]) => (
          <article key={title}>
            <Icon aria-hidden="true" size={21} />
            <h3>{title}</h3>
            <p>{description}</p>
          </article>
        ))}
      </div>
      <div className="data-note">
        <Info aria-hidden="true" size={18} />
        <p>
          這裡顯示的是公開登記資料快照。正式合作前，仍應確認合約、付款條件及實際履約能力。
        </p>
      </div>
    </div>
  );
}

function LoadingPanel({ message }: { message: string }) {
  return (
    <div className="loading-panel" role="status">
      <LoaderCircle className="spin" aria-hidden="true" size={28} />
      <div>
        <h2>{message}</h2>
        <p>系統正在整理 GCIS 登記資料並比對正式 Benchmark 快照。</p>
      </div>
    </div>
  );
}

function SearchResults({
  response,
  query,
  onSelect,
  onEdit,
}: {
  response: CompanySearchResponse;
  query: string;
  onSelect: (taxId: string) => void;
  onEdit: () => void;
}) {
  if (response.data.length === 0) {
    return (
      <div className="empty-panel" role="status" aria-live="polite" aria-atomic="true">
        <FileSearch aria-hidden="true" size={30} />
        <h2>沒有找到核准設立的公司</h2>
        <p>
          {query.trim() && <>「{query.trim()}」目前沒有搜尋結果。 </>}
          請嘗試完整公司名稱，或改用 8 碼統一編號查詢。
        </p>
        <div className="empty-actions">
          <button type="button" className="secondary-button" onClick={onEdit}>
            修改搜尋內容
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="search-results">
      <StaleDataNotice sources={[{ meta: response.meta }]} />
      <div className="section-heading row-heading">
        <div>
          <p className="section-kicker">找到 {response.data.length} 筆候選</p>
          <h2>選擇要查看的公司</h2>
        </div>
        <span className="result-scope">僅列核准設立</span>
      </div>
      <div className="candidate-list">
        {response.data.map((company) => (
          <CandidateRow
            company={company}
            key={company.tax_id}
            onSelect={onSelect}
          />
        ))}
      </div>
      <SourceLine meta={response.meta} />
    </div>
  );
}

function CandidateRow({
  company,
  onSelect,
}: {
  company: CompanySearchItem;
  onSelect: (taxId: string) => void;
}) {
  return (
    <button
      className="candidate-row"
      type="button"
      onClick={() => onSelect(company.tax_id)}
    >
      <span className="candidate-main">
        <span className="candidate-icon" aria-hidden="true">
          <Building2 size={20} />
        </span>
        <span>
          <strong>{company.name}</strong>
          <small>
            統編 <code>{company.tax_id}</code>
          </small>
        </span>
      </span>
      <span className="candidate-meta">
        <StatusPill status={company.status.description} code={company.status.code} />
        <span>{formatCurrency(company.registered_capital)}</span>
      </span>
      <ArrowRight className="row-arrow" aria-hidden="true" size={19} />
    </button>
  );
}

function CompanyPanel({
  response,
  onCompare,
  onReset,
}: {
  response: CompanyBizScoreResponse;
  onCompare: (company: CompanyData) => void;
  onReset: () => void;
}) {
  const { company: data } = response.data;
  const { meta } = response;
  return (
    <div className="company-panel">
      <StaleDataNotice
        sources={[{ label: data.name, meta }]}
      />
      {meta.warnings.length > 0 && (
        <DataQualityWarning partial={meta.partial} warnings={meta.warnings} />
      )}

      <article className="company-card">
        <header className="company-card-header">
          <div>
            <p className="section-kicker">公司登記摘要</p>
            <h2>{data.name}</h2>
            <p className="company-id">
              統一編號 <code>{data.tax_id}</code>
            </p>
          </div>
          <StatusPill
            status={data.status.description}
            code={data.status.code}
            prominent
          />
        </header>

        <nav className="report-navigation" aria-label="公司報告區段">
          <a href="#company-overview">01 基本資料</a>
          <a href="#company-score">02 BizScore</a>
          <a href="#company-ai">03 AI 查核</a>
          <a href="#business-title">04 營業項目</a>
        </nav>

        <dl className="company-facts" id="company-overview">
          <Fact label="核准設立日期" value={formatDate(data.established_at)} />
          <Fact
            label="成立年數"
            value={formatCompanyAge(
              data.established_at,
              meta.fetched_at,
              data.company_age_years,
            )}
          />
          <Fact
            label="資本總額"
            value={formatCurrency(data.capital.registered)}
          />
          <Fact
            label="實收資本額"
            value={formatCurrency(data.capital.paid_in)}
          />
          <Fact label="最後異動日期" value={formatDate(data.last_changed_at)} />
          <Fact
            label="登記機關"
            value={data.registration_authority ?? "資料未提供"}
          />
        </dl>

        <div className="identity-details">
          <DetailLine
            icon={UserRound}
            label="負責人"
            value={data.responsible_name}
          />
          <DetailLine icon={MapPin} label="公司地址" value={data.address} />
        </div>

        <div id="company-score"><BizScorePanel response={response} /></div>
        <div id="company-ai"><AIAnalysisPanel response={response} /></div>

        <BusinessItems company={data} />

        <footer className="company-card-footer">
          <SourceLine meta={meta} />
          <div className="company-card-actions">
            <button
              type="button"
              className="text-button compare-text-button"
              onClick={() => onCompare(data)}
            >
              <GitCompareArrows aria-hidden="true" size={16} />
              加入公司比較
            </button>
            <button type="button" className="text-button" onClick={onReset}>
              <RotateCcw aria-hidden="true" size={16} />
              查詢另一家公司
            </button>
          </div>
        </footer>
      </article>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="fact">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function DetailLine({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof MapPin;
  label: string;
  value: string | null;
}) {
  return (
    <div>
      <Icon aria-hidden="true" size={18} />
      <span>
        <small>{label}</small>
        <strong>{value ?? "資料未提供"}</strong>
      </span>
    </div>
  );
}

function BusinessItems({ company }: { company: CompanyData }) {
  return (
    <section className="business-section" aria-labelledby="business-title">
      <div className="business-heading">
        <div>
          <p className="section-kicker">登記營業範圍</p>
          <h3 id="business-title">營業項目</h3>
        </div>
        <span>{company.business_items.length} 項</span>
      </div>
      {company.business_items.length === 0 ? (
        <p className="business-empty" role="status">
          目前沒有可顯示的營業項目資料。
        </p>
      ) : (
        <ol className="business-list">
          {company.business_items.map((item) => (
            <li key={`${item.sequence}-${item.code}`}>
              <span className="business-sequence">{item.sequence}</span>
              <span>
                <code>{item.code}</code>
                <strong>{item.name ?? "項目名稱未提供"}</strong>
              </span>
            </li>
          ))}
        </ol>
      )}
      <p className="business-disclaimer">
        登記營業項目表示依法可經營的範圍，不代表目前各項業務皆有實際營收。
      </p>
    </section>
  );
}

function DataQualityWarning({
  partial,
  warnings,
}: {
  partial: boolean;
  warnings: string[];
}) {
  return (
    <div className="partial-warning" role="status">
      <AlertTriangle aria-hidden="true" size={21} />
      <div>
        <h2>{partial ? "目前顯示部分登記資料" : "這筆資料有需要留意的欄位"}</h2>
        <p>
          {partial
            ? "基本資料已取得，但營業項目服務暫時未完整回應。"
            : "上游資料存在缺漏或差異；主要值仍依既定合併規則顯示。"}
        </p>
        {warnings.length > 0 && (
          <details>
            <summary>查看技術說明</summary>
            <ul>
              {warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </div>
  );
}

function ErrorPanel({
  error,
  onRetry,
  onEdit,
}: {
  error: ApiError;
  onRetry: () => void;
  onEdit: () => void;
}) {
  return (
    <div className="error-panel" role="alert">
      <span className="error-icon" aria-hidden="true">
        <AlertTriangle size={25} />
      </span>
      <p className="section-kicker">查詢未完成 · {error.code}</p>
      <h2>{error.status === 404 ? "找不到這家公司" : "目前無法完成查詢"}</h2>
      <p>{error.message}</p>
      <div className="error-actions">
        {error.retryable && (
          <button type="button" onClick={onRetry}>
            <RotateCcw aria-hidden="true" size={17} />
            再試一次
          </button>
        )}
        <button type="button" className="secondary-button" onClick={onEdit}>
          修改查詢內容
        </button>
      </div>
    </div>
  );
}

function StatusPill({
  status,
  code,
  prominent = false,
}: {
  status: string | null;
  code: string | null;
  prominent?: boolean;
}) {
  const isActive = code === "01";
  return (
    <span
      className={`status-pill ${isActive ? "active" : "neutral"}${prominent ? " prominent" : ""}`}
    >
      {isActive ? (
        <CheckCircle2 aria-hidden="true" size={prominent ? 18 : 15} />
      ) : (
        <Info aria-hidden="true" size={prominent ? 18 : 15} />
      )}
      <span>{status ?? "狀態未提供"}</span>
      {code && <code>{code}</code>}
    </span>
  );
}

function SourceLine({ meta }: { meta: ResponseMeta }) {
  const freshness = meta.data_freshness ?? "live";
  const timeLabel =
    freshness === "stale_cache"
      ? "最近成功快照"
      : freshness === "fresh_cache"
        ? "短期快取快照"
        : "GCIS 資料取得時間";
  return (
    <div className="source-line">
      <Database aria-hidden="true" size={16} />
      <span>{meta.provider}</span>
      <span aria-hidden="true">·</span>
      <span className={`source-freshness ${freshness}`}>
        {timeLabel}{" "}
        <time dateTime={meta.fetched_at}>{formatDateTime(meta.fetched_at)}</time>
      </span>
    </div>
  );
}

function formatCurrency(value: number | null): string {
  if (value === null) return "資料未提供";
  return new Intl.NumberFormat("zh-TW", {
    style: "currency",
    currency: "TWD",
    maximumFractionDigits: 0,
  }).format(value);
}

function formatDate(value: string | null): string {
  return value ? value.replaceAll("-", "/") : "資料未提供";
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

function preferredScrollBehavior(): ScrollBehavior {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ? "auto"
    : "smooth";
}
