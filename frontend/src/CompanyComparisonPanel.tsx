import {
  type CSSProperties,
  type FormEvent,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  AlertTriangle,
  ArrowRight,
  Building2,
  CheckCircle2,
  Database,
  GitCompareArrows,
  Info,
  ListPlus,
  LoaderCircle,
  Plus,
  RotateCcw,
  Search,
  ShieldCheck,
  X,
} from "lucide-react";

import { ApiError, compareCompanies, searchCompanies } from "./api";
import AIComparisonPanel from "./AIComparisonPanel";
import { formatCompanyAge } from "./formatters";
import { noScoreLabel } from "./companyStatus";
import StaleDataNotice from "./StaleDataNotice";
import type {
  BizScoreDimension,
  CompanyComparisonItem,
  CompanyComparisonResponse,
  CompanySearchItem,
  CompanySearchResponse,
} from "./types";

export interface ComparisonSeed {
  taxId: string;
  name: string;
  nonce: number;
}

interface CompanyComparisonPanelProps {
  seed: ComparisonSeed | null;
  onViewCompany: (taxId: string) => void;
}

interface SelectedCompany {
  taxId: string;
  name: string | null;
}

type CandidateState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "results"; response: CompanySearchResponse }
  | { kind: "error"; error: ApiError };

type ComparisonState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "success"; response: CompanyComparisonResponse }
  | { kind: "error"; error: ApiError };

const MAX_COMPANIES = 3;
const MIN_COMPANIES = 2;

export default function CompanyComparisonPanel({
  seed,
  onViewCompany,
}: CompanyComparisonPanelProps) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<SelectedCompany[]>([]);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const [candidateState, setCandidateState] = useState<CandidateState>({
    kind: "idle",
  });
  const [comparisonState, setComparisonState] = useState<ComparisonState>({
    kind: "idle",
  });
  const candidateController = useRef<AbortController | null>(null);
  const candidateRequestIdentity = useRef(0);
  const comparisonController = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLElement>(null);

  useEffect(() => {
    return () => {
      candidateController.current?.abort();
      comparisonController.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!seed) return;
    setSelected((current) => {
      if (current.some((company) => company.taxId === seed.taxId)) return current;
      if (current.length >= MAX_COMPANIES) return current;
      return [...current, { taxId: seed.taxId, name: seed.name }];
    });
    setSelectionError(null);
    setComparisonState({ kind: "idle" });
  }, [seed]);

  useEffect(() => {
    if (
      comparisonState.kind === "success" ||
      comparisonState.kind === "error"
    ) {
      resultRef.current?.focus();
    }
  }, [comparisonState]);

  const isCandidateLoading = candidateState.kind === "loading";
  const isComparing = comparisonState.kind === "loading";
  const hasReachedLimit = selected.length >= MAX_COMPANIES;
  const normalizedQuery = query.trim();
  const queryIsTaxId = /^[0-9]{8}$/.test(normalizedQuery);

  const cancelCandidateRequest = () => {
    candidateRequestIdentity.current += 1;
    candidateController.current?.abort();
    candidateController.current = null;
  };

  const resetComparisonResult = () => {
    comparisonController.current?.abort();
    setComparisonState({ kind: "idle" });
  };

  const addSelectedCompany = (company: SelectedCompany) => {
    if (selected.some((item) => item.taxId === company.taxId)) {
      setSelectionError("這家公司已經在比較名單中。");
      inputRef.current?.focus();
      return;
    }
    if (selected.length >= MAX_COMPANIES) {
      setSelectionError("一次最多比較 3 家公司，請先移除一家公司。");
      return;
    }
    cancelCandidateRequest();
    setSelected((current) => [...current, company]);
    setQuery("");
    setSelectionError(null);
    setCandidateState({ kind: "idle" });
    resetComparisonResult();
    window.setTimeout(() => inputRef.current?.focus(), 0);
  };

  const removeSelectedCompany = (taxId: string) => {
    setSelected((current) =>
      current.filter((company) => company.taxId !== taxId),
    );
    setSelectionError(null);
    resetComparisonResult();
  };

  const submitCandidateQuery = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSelectionError(null);

    if (hasReachedLimit) {
      setSelectionError("比較名單已達 3 家上限，請先移除一家公司。");
      return;
    }
    if (!normalizedQuery) {
      setSelectionError("請輸入公司名稱或 8 碼統一編號。");
      inputRef.current?.focus();
      return;
    }
    if (/^[0-9０-９]+$/u.test(normalizedQuery)) {
      if (!queryIsTaxId) {
        setSelectionError("統一編號必須是 8 碼半形數字。");
        inputRef.current?.focus();
        return;
      }
      addSelectedCompany({ taxId: normalizedQuery, name: null });
      return;
    }
    if (normalizedQuery.length < 2) {
      setSelectionError("公司名稱至少需要 2 個字元。");
      inputRef.current?.focus();
      return;
    }

    cancelCandidateRequest();
    const controller = new AbortController();
    const requestIdentity = candidateRequestIdentity.current;
    candidateController.current = controller;
    setCandidateState({ kind: "loading" });
    try {
      const response = await searchCompanies(normalizedQuery, controller.signal);
      if (
        !controller.signal.aborted &&
        candidateController.current === controller &&
        candidateRequestIdentity.current === requestIdentity
      ) {
        setCandidateState({ kind: "results", response });
      }
    } catch (error) {
      if (
        controller.signal.aborted ||
        candidateController.current !== controller ||
        candidateRequestIdentity.current !== requestIdentity
      ) {
        return;
      }
      setCandidateState({
        kind: "error",
        error: toApiError(error, "搜尋候選公司時發生非預期錯誤。"),
      });
    }
  };

  const runComparison = async () => {
    if (selected.length < MIN_COMPANIES) {
      setSelectionError("請先加入至少 2 家公司再開始比較。");
      inputRef.current?.focus();
      return;
    }
    comparisonController.current?.abort();
    const controller = new AbortController();
    comparisonController.current = controller;
    setSelectionError(null);
    cancelCandidateRequest();
    setCandidateState({ kind: "idle" });
    setComparisonState({ kind: "loading" });
    const taxIds = selected.map((company) => company.taxId);
    try {
      const response = await compareCompanies(taxIds, controller.signal);
      if (!controller.signal.aborted) {
        setSelected(
          response.data.items.map((item) => ({
            taxId: item.company.tax_id,
            name: item.company.name,
          })),
        );
        setComparisonState({ kind: "success", response });
      }
    } catch (error) {
      if (controller.signal.aborted) return;
      setComparisonState({
        kind: "error",
        error: toApiError(error, "公司比較時發生非預期錯誤。"),
      });
    }
  };

  const clearSelection = () => {
    cancelCandidateRequest();
    comparisonController.current?.abort();
    setQuery("");
    setSelected([]);
    setSelectionError(null);
    setCandidateState({ kind: "idle" });
    setComparisonState({ kind: "idle" });
    window.setTimeout(() => inputRef.current?.focus(), 0);
  };

  const focusSelector = () => {
    inputRef.current?.focus();
    document.getElementById("comparison-selector")?.scrollIntoView({
      behavior: preferredScrollBehavior(),
      block: "start",
    });
  };

  const comparisonStep =
    comparisonState.kind === "success"
      ? 3
      : comparisonState.kind === "loading" || selected.length >= MIN_COMPANIES
        ? 2
        : 1;

  return (
    <>
      <section
        className="search-stage comparison-search-stage"
        id="comparison-selector"
        aria-labelledby="comparison-page-title"
      >
        <div className="thesis comparison-thesis">
          <p className="eyebrow">2–3 家公司公開資料並列查核</p>
          <h1 id="comparison-page-title">把公司放在同一張資料桌上。</h1>
          <p>
            依選取順序並列登記資料與 BizScore；不同產業的 PR 只代表各自在同業中的位置，不判定勝負或推薦合作對象。
          </p>
        </div>

        <form
          className="search-form comparison-search-form"
          onSubmit={submitCandidateQuery}
          noValidate
        >
          <label htmlFor="comparison-query">搜尋並加入公司</label>
          <div className="search-control">
            <Search className="search-icon" aria-hidden="true" size={21} />
            <input
              ref={inputRef}
              id="comparison-query"
              value={query}
              onChange={(event) => {
                cancelCandidateRequest();
                setQuery(event.target.value);
                setCandidateState({ kind: "idle" });
                if (selectionError) setSelectionError(null);
              }}
              aria-describedby={
                selectionError
                  ? "comparison-query-help comparison-query-error"
                  : "comparison-query-help"
              }
              aria-invalid={selectionError ? "true" : "false"}
              autoComplete="off"
              disabled={hasReachedLimit || isComparing}
              placeholder="例如：宏碁、20828393"
            />
            <button
              type="submit"
              disabled={hasReachedLimit || isCandidateLoading || isComparing}
            >
              {isCandidateLoading ? (
                <LoaderCircle className="spin" aria-hidden="true" size={19} />
              ) : queryIsTaxId ? (
                <Plus aria-hidden="true" size={19} />
              ) : (
                <Search aria-hidden="true" size={19} />
              )}
              {isCandidateLoading
                ? "搜尋中"
                : queryIsTaxId
                  ? "加入統編"
                  : "搜尋候選"}
            </button>
          </div>
          <div className="form-support">
            <p id="comparison-query-help">
              名稱搜尋後再選公司；輸入 8 碼統編可直接加入，送出比較時確認資料。
            </p>
            {selectionError && (
              <p id="comparison-query-error" className="field-error" role="alert">
                {selectionError}
              </p>
            )}
          </div>
        </form>

        <CandidatePicker
          state={candidateState}
          selectedTaxIds={new Set(selected.map((company) => company.taxId))}
          onAdd={(company) =>
            addSelectedCompany({ taxId: company.tax_id, name: company.name })
          }
          onRetry={() => {
            const form = inputRef.current?.form;
            form?.requestSubmit();
          }}
          onEditQuery={() => {
            inputRef.current?.focus();
            document.getElementById("comparison-selector")?.scrollIntoView({
              behavior: preferredScrollBehavior(),
              block: "start",
            });
          }}
        />

        <div className="comparison-selection-shell">
          <div className="comparison-selection-heading">
            <div>
              <p className="section-kicker">比較名單</p>
              <h2>{selected.length} / {MAX_COMPANIES} 家公司</h2>
            </div>
            {selected.length > 0 && (
              <button
                type="button"
                className="comparison-clear-button"
                onClick={clearSelection}
                disabled={isComparing}
              >
                <RotateCcw aria-hidden="true" size={15} />
                清空名單
              </button>
            )}
          </div>

          <SelectionSlots
            selected={selected}
            disabled={isComparing}
            onRemove={removeSelectedCompany}
          />

          <div className="comparison-submit-row">
            <p aria-live="polite">
              {selected.length < MIN_COMPANIES
                ? `再加入 ${MIN_COMPANIES - selected.length} 家即可開始比較。`
                : selected.length === MAX_COMPANIES
                  ? "已選滿 3 家；結果會依 A、B、C 順序並列。"
                  : "現在可以比較，也可以再加入第 3 家公司。"}
            </p>
            <button
              type="button"
              className="comparison-submit-button"
              onClick={() => void runComparison()}
              disabled={selected.length < MIN_COMPANIES || isComparing}
            >
              {isComparing ? (
                <LoaderCircle className="spin" aria-hidden="true" size={19} />
              ) : (
                <GitCompareArrows aria-hidden="true" size={19} />
              )}
              {isComparing
                ? "正在建立比較表"
                : `開始比較 ${selected.length} 家公司`}
            </button>
          </div>
        </div>

        <ComparisonRail currentStep={comparisonStep} />
      </section>

      <section
        ref={resultRef}
        className="result-stage comparison-result-stage"
        tabIndex={-1}
        aria-busy={isComparing}
      >
        {comparisonState.kind === "idle" && <ComparisonIdlePanel />}
        {comparisonState.kind === "loading" && <ComparisonLoadingPanel />}
        {comparisonState.kind === "error" && (
          <ComparisonErrorPanel
            error={comparisonState.error}
            onEdit={focusSelector}
            onRetry={() => void runComparison()}
          />
        )}
        {comparisonState.kind === "success" && (
          <ComparisonReport
            response={comparisonState.response}
            onEdit={focusSelector}
            onViewCompany={onViewCompany}
          />
        )}
      </section>
    </>
  );
}

function CandidatePicker({
  state,
  selectedTaxIds,
  onAdd,
  onRetry,
  onEditQuery,
}: {
  state: CandidateState;
  selectedTaxIds: Set<string>;
  onAdd: (company: CompanySearchItem) => void;
  onRetry: () => void;
  onEditQuery: () => void;
}) {
  if (state.kind === "idle") return null;
  if (state.kind === "loading") {
    return (
      <div className="comparison-candidate-status" role="status">
        <LoaderCircle className="spin" aria-hidden="true" size={18} />
        正在尋找可加入比較的公司…
      </div>
    );
  }
  if (state.kind === "error") {
    return (
      <div className="comparison-candidate-error" role="alert">
        <AlertTriangle aria-hidden="true" size={18} />
        <span>{state.error.message}</span>
        {state.error.retryable && (
          <button type="button" onClick={onRetry}>再試一次</button>
        )}
      </div>
    );
  }
  if (state.response.data.length === 0) {
    return (
      <div
        className="comparison-candidate-status"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        <Info aria-hidden="true" size={18} />
        <span>搜尋完成，共 0 筆候選公司。請改用完整名稱或 8 碼統編。</span>
        <button
          type="button"
          className="comparison-candidate-status-button"
          onClick={onEditQuery}
        >
          修改搜尋詞
        </button>
      </div>
    );
  }
  return (
    <>
      <StaleDataNotice sources={[{ meta: state.response.meta }]} />
      <div className="comparison-candidate-picker" aria-label="候選公司">
      <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        搜尋完成，共 {state.response.data.length} 筆候選公司。
      </p>
      <div className="comparison-candidate-topline">
        <strong>選擇要加入的公司</strong>
        <span>{state.response.data.length} 筆候選</span>
      </div>
      <ul>
        {state.response.data.map((company) => {
          const isSelected = selectedTaxIds.has(company.tax_id);
          return (
            <li key={company.tax_id}>
              <button
                type="button"
                onClick={() => onAdd(company)}
                disabled={isSelected}
                aria-label={
                  isSelected
                    ? `${company.name}（${company.tax_id}）已在比較名單`
                    : `加入${company.name}（${company.tax_id}）`
                }
              >
                <span className="candidate-icon" aria-hidden="true">
                  <Building2 size={18} />
                </span>
                <span>
                  <strong>{company.name}</strong>
                  <small>統編 <code>{company.tax_id}</code></small>
                </span>
                <span className="comparison-candidate-action">
                  {isSelected ? (
                    <CheckCircle2 aria-hidden="true" size={17} />
                  ) : (
                    <Plus aria-hidden="true" size={17} />
                  )}
                  {isSelected ? "已加入" : "加入"}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      </div>
    </>
  );
}

function preferredScrollBehavior(): ScrollBehavior {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ? "auto"
    : "smooth";
}

function SelectionSlots({
  selected,
  disabled,
  onRemove,
}: {
  selected: SelectedCompany[];
  disabled: boolean;
  onRemove: (taxId: string) => void;
}) {
  return (
    <ol className="comparison-slots" aria-label="已選比較公司，順序即比較順序">
      {Array.from({ length: MAX_COMPANIES }, (_, index) => {
        const company = selected[index];
        const letter = String.fromCharCode(65 + index);
        if (!company) {
          return (
            <li className="comparison-slot empty" key={`empty-${letter}`}>
              <span className="comparison-letter" aria-hidden="true">{letter}</span>
              <span>
                <strong>尚未選擇</strong>
                <small>{index < MIN_COMPANIES ? "必選公司" : "選填第 3 家"}</small>
              </span>
            </li>
          );
        }
        return (
          <li className="comparison-slot selected" key={company.taxId}>
            <span className="comparison-letter" aria-hidden="true">{letter}</span>
            <span className="comparison-slot-company">
              <strong>{company.name ?? `統編 ${company.taxId}`}</strong>
              <small>
                {company.name ? "統編 " : "公司名稱將在比較時確認 "}
                <code>{company.taxId}</code>
              </small>
            </span>
            <button
              type="button"
              onClick={() => onRemove(company.taxId)}
              disabled={disabled}
              aria-label={`從比較名單移除${company.name ?? company.taxId}`}
            >
              <X aria-hidden="true" size={17} />
            </button>
          </li>
        );
      })}
    </ol>
  );
}

function ComparisonRail({ currentStep }: { currentStep: number }) {
  const steps = ["加入 2–3 家", "建立資料對照", "檢視差異與限制"];
  return (
    <ol className="verification-rail comparison-rail" aria-label="公司比較進度">
      {steps.map((step, index) => {
        const stepNumber = index + 1;
        const isActive = stepNumber <= currentStep;
        return (
          <li className={isActive ? "active" : ""} key={step}>
            <span aria-hidden="true">
              {isActive ? <CheckCircle2 size={18} /> : stepNumber}
            </span>
            <strong>{step}</strong>
          </li>
        );
      })}
    </ol>
  );
}

function ComparisonIdlePanel() {
  const facts = [
    [ListPlus, "依選取順序", "A、B、C 只代表加入順序，不是名次。"],
    [GitCompareArrows, "同欄位並列", "分數、五構面與登記資料使用相同定義。"],
    [ShieldCheck, "先看限制", "缺值、暫定分數與跨產業基準不會被隱藏。"],
  ] as const;
  return (
    <div className="comparison-idle-panel">
      <div className="section-heading">
        <p className="section-kicker">PK 比較表會整理成</p>
        <h2>一份不替你下結論的公開資料對照簿</h2>
      </div>
      <div className="comparison-idle-grid">
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
          請先加入 2 家公司。比較表只並列政府公開資料，不會標示贏家、推薦合作對象或推測付款與履約能力。
        </p>
      </div>
    </div>
  );
}

function ComparisonLoadingPanel() {
  return (
    <div className="loading-panel comparison-loading-panel" role="status">
      <LoaderCircle className="spin" aria-hidden="true" size={28} />
      <div>
        <h2>正在建立公司 PK 比較表…</h2>
        <p>逐家公司整理 GCIS、BizScore 與正式 Benchmark，結果會一次完整顯示。</p>
      </div>
    </div>
  );
}

function ComparisonErrorPanel({
  error,
  onEdit,
  onRetry,
}: {
  error: ApiError;
  onEdit: () => void;
  onRetry: () => void;
}) {
  return (
    <div className="error-panel comparison-error-panel" role="alert">
      <span className="error-icon" aria-hidden="true">
        <AlertTriangle size={25} />
      </span>
      <p className="section-kicker">比較未完成 · {error.code}</p>
      <h2>{error.status === 404 ? "比較名單中有公司查無資料" : "目前無法完成公司比較"}</h2>
      <p>{error.message}</p>
      <p className="comparison-atomic-note">
        為避免混用不完整結果，這次沒有顯示其他公司的部分比較資料；已選名單仍保留。
      </p>
      <div className="error-actions">
        {error.retryable && (
          <button type="button" onClick={onRetry}>
            <RotateCcw aria-hidden="true" size={17} />
            重新比較
          </button>
        )}
        <button type="button" className="secondary-button" onClick={onEdit}>
          調整比較名單
        </button>
      </div>
    </div>
  );
}

function ComparisonReport({
  response,
  onEdit,
  onViewCompany,
}: {
  response: CompanyComparisonResponse;
  onEdit: () => void;
  onViewCompany: (taxId: string) => void;
}) {
  const { items, context, disclaimer } = response.data;
  const countStyle = {
    "--comparison-count": items.length,
  } as CSSProperties;
  const rows = buildComparisonRows(response);

  return (
    <article className="comparison-report" aria-labelledby="comparison-report-title">
      <header className="comparison-report-header">
        <div>
          <p className="section-kicker">公開登記資料 PK 對照</p>
          <h2 id="comparison-report-title">{items.length} 家公司比較表</h2>
          <p>公司 A、B、C 依你的加入順序排列；本表不計算名次或勝負。</p>
        </div>
        <button type="button" className="comparison-edit-button" onClick={onEdit}>
          <ListPlus aria-hidden="true" size={17} />
          調整比較名單
        </button>
      </header>

      <div className="comparison-context-bar">
        <span>
          <Database aria-hidden="true" size={15} />
          基準日 {formatDate(context.benchmark_as_of)}
        </span>
        <span>{context.same_primary_industry ? "相同主要產業" : "不同主要產業"}</span>
        <span>依選取順序 · 不排名</span>
      </div>

      <ComparisonNotices response={response} />

      <div className="comparison-ledger-desktop" style={countStyle}>
        <table className="comparison-ledger-table">
          <caption className="sr-only">
            {items.length} 家公司的公開登記資料與 BizScore 並列表
          </caption>
          <thead>
            <tr>
              <th scope="col" className="comparison-metric-column">
                比較欄位
              </th>
              {items.map((item, index) => (
                <th scope="col" key={item.company.tax_id}>
                  <ComparisonCompanyHeading
                    item={item}
                    letter={String.fromCharCode(65 + index)}
                    onViewCompany={onViewCompany}
                  />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key}>
                <th scope="row">
                  <strong>{row.label}</strong>
                  <small>{row.help}</small>
                </th>
                {items.map((item) => (
                  <td key={`${row.key}-${item.company.tax_id}`}>
                    {row.render(item)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="comparison-ledger-mobile">
        {items.map((item, index) => (
          <section
            className="comparison-mobile-card"
            key={item.company.tax_id}
            aria-labelledby={`comparison-mobile-${item.company.tax_id}`}
          >
            <ComparisonCompanyHeading
              item={item}
              letter={String.fromCharCode(65 + index)}
              onViewCompany={onViewCompany}
              headingId={`comparison-mobile-${item.company.tax_id}`}
            />
            <dl>
              {rows.map((row) => (
                <div key={row.key}>
                  <dt>
                    <strong>{row.label}</strong>
                    <small>{row.help}</small>
                  </dt>
                  <dd>{row.render(item)}</dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>

      {response.meta.warnings.length > 0 && (
        <details className="comparison-warning-details">
          <summary>查看 {response.meta.warnings.length} 項比較限制與來源提醒</summary>
          <ul>
            {response.meta.warnings.map((warning, index) => (
              <li key={`${warning.code}-${warning.tax_id ?? "global"}-${index}`}>
                <code>{warning.code}</code>
                <span>{warning.message}</span>
                {warning.tax_id && <small>統編 {warning.tax_id}</small>}
              </li>
            ))}
          </ul>
        </details>
      )}

      <AIComparisonPanel comparison={response} />

      <div className="comparison-disclaimer">
        <ShieldCheck aria-hidden="true" size={20} />
        <p><strong>這不是信用評等或合作建議。</strong>{disclaimer}</p>
      </div>

      <footer className="comparison-report-footer">
        <span>Benchmark <code>{context.benchmark_catalog_version}</code></span>
        <span>產生時間 {formatDateTime(response.meta.generated_at)}</span>
      </footer>
    </article>
  );
}

function ComparisonNotices({ response }: { response: CompanyComparisonResponse }) {
  const { context } = response.data;
  const flags: string[] = [];
  if (
    response.data.items.some(
      (item) => (item.source_meta.data_freshness ?? "live") === "stale_cache",
    )
  ) {
    flags.push("含最近成功資料快照");
  }
  if (response.meta.has_partial_source_data) flags.push("部分政府來源資料");
  if (response.meta.has_provisional_scores) flags.push("含暫定 BizScore");
  if (response.meta.has_unscored_companies) flags.push("含無數字總分公司");

  return (
    <div className="comparison-notice-stack">
      <StaleDataNotice
        sources={response.data.items.map((item) => ({
          label: `${item.company.name}（${item.company.tax_id}）`,
          meta: item.source_meta,
        }))}
      />
      {context.peer_comparison_scope === "different_industry_snapshots" && (
        <div className="comparison-notice warning" role="status">
          <AlertTriangle aria-hidden="true" size={19} />
          <div>
            <strong>這些公司的 PR 來自不同產業快照</strong>
            <p>PR 只能解讀為各自在所屬同業中的位置，不能排成同一份跨產業名次。</p>
          </div>
        </div>
      )}
      {context.peer_comparison_scope === "unavailable" && (
        <div className="comparison-notice neutral" role="status">
          <Info aria-hidden="true" size={19} />
          <div>
            <strong>至少一家公司缺少可用的同業位置</strong>
            <p>缺值會顯示為「資料未提供」，不會補成 PR 0。</p>
          </div>
        </div>
      )}
      {flags.length > 0 && (
        <div className="comparison-flag-row" aria-label="本次比較資料狀態">
          {flags.map((flag) => <span key={flag}>{flag}</span>)}
        </div>
      )}
    </div>
  );
}

function ComparisonCompanyHeading({
  item,
  letter,
  onViewCompany,
  headingId,
}: {
  item: CompanyComparisonItem;
  letter: string;
  onViewCompany: (taxId: string) => void;
  headingId?: string;
}) {
  return (
    <div className="comparison-company-heading">
      <span className="comparison-letter" aria-hidden="true">{letter}</span>
      <div>
        <h3 id={headingId}>{item.company.name}</h3>
        <p>統編 <code>{item.company.tax_id}</code></p>
      </div>
      <ComparisonStatus item={item} />
      <button type="button" onClick={() => onViewCompany(item.company.tax_id)}>
        查看單家公司
        <ArrowRight aria-hidden="true" size={15} />
      </button>
    </div>
  );
}

interface ComparisonRow {
  key: string;
  label: string;
  help: string;
  render: (item: CompanyComparisonItem) => ReactNode;
}

function buildComparisonRows(response: CompanyComparisonResponse): ComparisonRow[] {
  const peerScope = response.data.context.peer_comparison_scope;
  return [
    {
      key: "bizscore",
      label: "BizScore",
      help: "固定規則，不由 AI 判分",
      render: (item) => <ScoreValue item={item} />,
    },
    {
      key: "coverage",
      label: "資料覆蓋率",
      help: "可計分構面占比",
      render: (item) => (
        <ComparisonValue
          value={`${formatPercent(item.metrics.bizscore_coverage)}%`}
          detail={item.bizscore.provisional ? "暫定結果" : "依 API 原值顯示"}
          tone={item.bizscore.provisional ? "warning" : "default"}
        />
      ),
    },
    {
      key: "registration-status",
      label: "登記狀態",
      help: "GCIS 官方登記狀態",
      render: (item) => (
        <ComparisonValue
          value={item.metrics.registration_status ?? "資料未提供"}
          detail={item.company.status.code ? `代碼 ${item.company.status.code}` : "狀態代碼未提供"}
          missing={item.metrics.registration_status === null}
        />
      ),
    },
    {
      key: "company-age",
      label: "成立年資",
      help: `計算至 ${formatDate(response.data.context.benchmark_as_of)}`,
      render: (item) => (
        <ComparisonValue
          value={formatCompanyAge(
            item.company.established_at,
            response.data.context.benchmark_as_of,
            item.metrics.company_age_years,
          )}
          detail={formatDimensionScore(findDimension(item, "company_age"))}
          missing={item.metrics.company_age_years === null}
        />
      ),
    },
    {
      key: "capital",
      label: "登記資本額",
      help: "不是現金、營收或清償能力",
      render: (item) => (
        <ComparisonValue
          value={formatCurrency(
            item.metrics.registered_capital,
            item.company.capital.currency,
          )}
          detail={formatDimensionScore(findDimension(item, "registered_capital_scale"))}
          missing={item.metrics.registered_capital === null}
        />
      ),
    },
    {
      key: "last-change",
      label: "最後登記異動",
      help: "只代表最近一次登記日期",
      render: (item) => (
        <ComparisonValue
          value={formatOptionalDate(item.metrics.last_changed_at)}
          detail={formatDimensionScore(findDimension(item, "registration_change_recency"))}
          missing={item.metrics.last_changed_at === null}
        />
      ),
    },
    {
      key: "industry",
      label: "主要產業",
      help: "依登記營業項目分類",
      render: (item) => {
        const industry = item.bizscore.industry.primary_group;
        return (
          <ComparisonValue
            value={industry ? `${industry.category_code}｜${industry.category_name}` : "資料未提供"}
            detail={item.bizscore.benchmark.snapshot_version ?? "無可用產業快照"}
            missing={!industry}
          />
        );
      },
    },
    {
      key: "peer-index",
      label: "同業綜合位置",
      help: "年資與資本 PR 的綜合指標",
      render: (item) => (
        <ComparisonValue
          value={item.metrics.peer_index === null ? "資料未提供" : `指標 ${formatDecimal(item.metrics.peer_index)}`}
          detail={
            peerScope === "different_industry_snapshots"
              ? "僅限各自同業群組"
              : peerScope === "same_industry_snapshot"
                ? "使用同一產業快照"
                : "未納入共同比較"
          }
          tone={peerScope === "different_industry_snapshots" ? "warning" : "default"}
          missing={item.metrics.peer_index === null}
        />
      ),
    },
    ...dimensionRows(),
    {
      key: "source-quality",
      label: "來源完整性",
      help: "GCIS A1／A3 與計分提醒",
      render: (item) => {
        const freshness = item.source_meta.data_freshness ?? "live";
        const isStale = freshness === "stale_cache";
        const value = isStale
          ? "最近成功快照"
          : freshness === "fresh_cache"
            ? "短期快取"
            : item.source_meta.partial
              ? "部分資料"
              : "即時來源";
        const detail = isStale
          ? `快照 ${formatDateTime(item.source_meta.fetched_at)}`
          : item.source_meta.warnings.length > 0
            ? `${item.source_meta.warnings.length} 項來源提醒`
            : item.bizscore.warnings.length > 0
              ? `${item.bizscore.warnings.length} 項計分提醒`
              : `取得 ${formatDateTime(item.source_meta.fetched_at)}`;
        return (
          <ComparisonValue
            value={value}
            detail={detail}
            tone={isStale || item.source_meta.partial ? "warning" : "default"}
          />
        );
      },
    },
  ];
}

function dimensionRows(): ComparisonRow[] {
  const dimensions = [
    ["dimension-registration", "登記狀態分數", "registration_status"],
    ["dimension-age", "成立年資分數", "company_age"],
    ["dimension-capital", "資本規模分數", "registered_capital_scale"],
    ["dimension-change", "登記異動分數", "registration_change_recency"],
    ["dimension-peer", "同業位置分數", "peer_relative_position"],
  ] as const;
  return dimensions.map(([key, label, dimensionKey]) => ({
    key,
    label,
    help: "BizScore v1 構面",
    render: (item: CompanyComparisonItem) => {
      const dimension = findDimension(item, dimensionKey);
      return (
        <ComparisonValue
          value={formatDimensionScore(dimension)}
          detail={dimension?.available ? "已納入覆蓋率" : "未計分不等於 0 分"}
          missing={!dimension?.available}
        />
      );
    },
  }));
}

function ScoreValue({ item }: { item: CompanyComparisonItem }) {
  const score = item.metrics.bizscore;
  return (
    <div className={`comparison-score-value${item.metrics.bizscore_provisional ? " provisional" : ""}`}>
      <strong aria-label={score === null ? "沒有數字總分" : `BizScore ${score} 分`}>
        {score === null ? "—" : score}
      </strong>
      <span>/ 100</span>
      <small>{item.bizscore.band ?? noScoreLabel(item.company.status)}</small>
      {item.metrics.bizscore_provisional && <em>暫定分數</em>}
      {item.bizscore.status_cap !== null && item.bizscore.status_cap < 100 && (
        <small>登記狀態上限 {item.bizscore.status_cap} 分</small>
      )}
    </div>
  );
}

function ComparisonStatus({ item }: { item: CompanyComparisonItem }) {
  const active = item.company.status.code === "01";
  return (
    <span className={`comparison-company-status ${active ? "active" : "neutral"}`}>
      {active ? (
        <CheckCircle2 aria-hidden="true" size={15} />
      ) : (
        <Info aria-hidden="true" size={15} />
      )}
      {item.company.status.description ?? "狀態未提供"}
    </span>
  );
}

function ComparisonValue({
  value,
  detail,
  missing = false,
  tone = "default",
}: {
  value: string;
  detail: string;
  missing?: boolean;
  tone?: "default" | "warning";
}) {
  return (
    <div className={`comparison-value${missing ? " missing" : ""}${tone === "warning" ? " warning" : ""}`}>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function findDimension(
  item: CompanyComparisonItem,
  key: string,
): BizScoreDimension | undefined {
  return item.bizscore.dimensions.find((dimension) => dimension.key === key);
}

function formatDimensionScore(dimension: BizScoreDimension | undefined): string {
  if (!dimension || !dimension.available || dimension.score === null) {
    return `資料不足 / ${dimension?.max_score ?? "—"}`;
  }
  return `${dimension.score} / ${dimension.max_score}`;
}

function formatCurrency(value: number | null, currency = "TWD"): string {
  if (value === null) return "資料未提供";
  try {
    return new Intl.NumberFormat("zh-TW", {
      style: "currency",
      currency,
      maximumFractionDigits: 0,
    }).format(value);
  } catch {
    return `${new Intl.NumberFormat("zh-TW").format(value)} ${currency}`;
  }
}

function formatDate(value: string): string {
  return value.replaceAll("-", "/");
}

function formatOptionalDate(value: string | null): string {
  return value ? formatDate(value) : "資料未提供";
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

function formatPercent(value: number): string {
  const percent = value * 100;
  return new Intl.NumberFormat("zh-TW", {
    maximumFractionDigits: Number.isInteger(percent) ? 0 : 1,
  }).format(percent);
}

function formatDecimal(value: number): string {
  return new Intl.NumberFormat("zh-TW", { maximumFractionDigits: 1 }).format(value);
}

function toApiError(error: unknown, fallbackMessage: string): ApiError {
  if (error instanceof ApiError) return error;
  return new ApiError(0, {
    code: "UNEXPECTED_ERROR",
    message: fallbackMessage,
    retryable: true,
  });
}
