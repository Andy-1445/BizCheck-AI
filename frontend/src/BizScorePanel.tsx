import {
  AlertTriangle,
  CheckCircle2,
  Database,
  Info,
  ShieldCheck,
} from "lucide-react";

import { formatCompanyAge } from "./formatters";
import { noScoreLabel } from "./companyStatus";
import {
  buildBizScoreExplanation,
  type ScoreDimensionExplanation,
} from "./scoreExplanation";
import type { CompanyBizScoreResponse } from "./types";

interface BizScorePanelProps {
  response: CompanyBizScoreResponse;
}

export default function BizScorePanel({ response }: BizScorePanelProps) {
  const { company, bizscore } = response.data;
  const companyAgeAsOf = bizscore.as_of;
  const explanation = buildBizScoreExplanation(response);
  const coverageValue = bizscore.coverage * 100;
  const coveragePercent = formatCoveragePercent(bizscore.coverage);
  const scoreLabel = bizscore.score === null ? "尚無法計分" : `${bizscore.score} 分`;
  const industry = bizscore.industry.primary_group;
  const peer = bizscore.peer_benchmark;

  return (
    <section
      className={`bizscore-section${bizscore.provisional ? " provisional" : ""}`}
      aria-labelledby="bizscore-title"
    >
      <header className="bizscore-heading">
        <div>
          <p className="section-kicker">公開資料體質指標</p>
          <h3 id="bizscore-title">BizScore v{bizscore.version}</h3>
          <p>
            依公司登記資料與同業基準計算，讓每一分都能回到公開欄位核對。
          </p>
        </div>
        <span className="score-as-of">資料基準日 {formatDate(bizscore.as_of)}</span>
      </header>

      <div className="score-ledger">
        <div className="score-docket">
          <div className="docket-topline">
            <span>企業體質評分單</span>
            <code>PUBLIC-DATA / V{bizscore.version}</code>
          </div>

          <div className="score-readout">
            <div
              className="score-number"
              aria-label={`BizScore ${scoreLabel}，滿分 100 分`}
              aria-describedby="bizscore-explanation-summary"
            >
              <strong>{bizscore.score ?? "—"}</strong>
              <span>/ 100</span>
            </div>
            <div className="score-verdict">
              <span className="verdict-label">公開資料觀察</span>
              <strong>{bizscore.band ?? noScoreLabel(company.status)}</strong>
              {bizscore.status_cap !== null && bizscore.status_cap < 100 && (
                <small>登記狀態使總分上限為 {bizscore.status_cap} 分</small>
              )}
            </div>
          </div>

          <div className="coverage-block">
            <div className="coverage-label">
              <span>可計分資料覆蓋率</span>
              <strong>{coveragePercent}%</strong>
            </div>
            <progress
              aria-label={`可計分資料覆蓋率 ${coveragePercent}%`}
              max={100}
              value={coverageValue}
            />
            <p>
              {bizscore.score === null
                ? "目前不顯示數字總分；請先確認缺漏構面與公司登記狀態。"
                : bizscore.provisional
                  ? "這是暫定結果；請查看右側說明確認構面缺漏或來源資料提醒。"
                  : "五個構面資料皆可用；仍需搭配合約、付款與實際履約查核。"}
            </p>
          </div>

          {bizscore.provisional && bizscore.score !== null && (
            <div className="provisional-stamp" role="status">
              <AlertTriangle aria-hidden="true" size={18} />
              <span>暫定分數，不等同完整 {bizscore.score} 分</span>
            </div>
          )}
        </div>

        <section
          className="dimension-sheet"
          id="bizscore-explanation"
          aria-labelledby="bizscore-explanation-title"
        >
          <header className="dimension-heading score-explanation-heading">
            <div>
              <span>BizScore v1 分數解釋</span>
              <h4 id="bizscore-explanation-title">{explanation.headline}</h4>
            </div>
            <span className="fixed-rule-label">
              <ShieldCheck aria-hidden="true" size={15} />
              固定規則｜非 AI 產生
            </span>
          </header>

          <div className={`score-explanation-copy ${explanation.state}`}>
            <span className="score-explanation-state">
              {explanation.stateLabel}
            </span>
            <p id="bizscore-explanation-summary">{explanation.summary}</p>
            <p className="score-explanation-composition">
              {explanation.composition}
            </p>
            {explanation.normalizationNote && (
              <p className="score-explanation-normalization">
                <Info aria-hidden="true" size={16} />
                <span>{explanation.normalizationNote}</span>
              </p>
            )}
            {explanation.capNotice && (
              <p className="score-explanation-cap">
                <AlertTriangle aria-hidden="true" size={16} />
                <span>{explanation.capNotice}</span>
              </p>
            )}
            <p className="score-explanation-boundary">
              這段說明只翻譯 API 已回傳的分數與公開資料，不使用 AI，也不重新計分。
            </p>
          </div>

          <div className="dimension-breakdown-heading">
            <div>
              <span>五構面逐項說明</span>
              <strong>{explanation.dimensions.length} 個構面</strong>
            </div>
            <small>主要原因直接顯示；完整公開資料可逐項展開</small>
          </div>
          <ol className="dimension-list">
            {explanation.dimensions.map((dimension, index) => (
              <DimensionRow
                explanation={dimension}
                key={`${dimension.dimension.key}-${index}`}
              />
            ))}
          </ol>
        </section>
      </div>

      <section className="peer-section" aria-labelledby="peer-title">
        <div className="peer-heading">
          <div>
            <p className="section-kicker">同業相對位置</p>
            <h4 id="peer-title">
              {industry
                ? `${industry.category_code}｜${industry.category_name}`
                : "目前無法建立同業比較群組"}
            </h4>
          </div>
          {peer.sample_size > 0 && (
            <span className="sample-badge">
              <Database aria-hidden="true" size={15} />
              {formatInteger(peer.sample_size)} 家樣本
            </span>
          )}
        </div>

        {peer.dimension.available ? (
          <div className="peer-grid">
            <PeerMetric
              label="成立年資 PR"
              value={peer.age_percentile}
              companyValue={
                `公司 ${formatCompanyAge(
                  company.established_at,
                  companyAgeAsOf,
                  company.company_age_years,
                )}`
              }
              medianValue={
                peer.company_age_median === null
                  ? "同業中位數未提供"
                  : `同業中位數 ${formatDecimal(peer.company_age_median)} 年`
              }
            />
            <PeerMetric
              label="登記資本 PR"
              value={peer.capital_percentile}
              companyValue={`公司 ${formatCurrency(company.capital.registered)}`}
              medianValue={
                peer.registered_capital_median === null
                  ? "同業中位數未提供"
                  : `同業中位數 ${formatCurrency(peer.registered_capital_median)}`
              }
            />
            <PeerMetric
              label="同業綜合位置"
              composite
              value={peer.peer_index}
              companyValue="年資 PR 與資本 PR 各占一半的平均值"
              medianValue="50 為刻度中點，不是綜合排名的中位數"
            />
          </div>
        ) : (
          <div className="peer-empty" role="status">
            <Info aria-hidden="true" size={20} />
            <div>
              <strong>目前沒有足夠資料計算同業位置</strong>
              <p>
                可能原因包含產業無法分類、樣本不足，或公司年資／資本欄位缺漏。
              </p>
            </div>
          </div>
        )}

        <div className="benchmark-reference">
          <span>
            Benchmark：<code>{bizscore.benchmark.catalog_version}</code>
          </span>
          <span>基準日：{formatDate(bizscore.benchmark.catalog_as_of)}</span>
          {bizscore.benchmark.snapshot_version && (
            <span>
              快照：<code>{bizscore.benchmark.snapshot_version}</code>
            </span>
          )}
        </div>
      </section>

      {(bizscore.warnings.length > 0 || bizscore.industry.warnings.length > 0) && (
        <details className="score-warnings">
          <summary>查看本次計分限制</summary>
          <ul>
            {[...bizscore.warnings, ...bizscore.industry.warnings].map(
              (warning, index) => (
                <li key={`${warning}-${index}`}>{warning}</li>
              ),
            )}
          </ul>
        </details>
      )}

      <div className="score-disclaimer">
        <ShieldCheck aria-hidden="true" size={19} />
        <p>
          <strong>BizScore 不是信用評等。</strong>
          分數僅整理政府公開登記資料，不代表履約能力、付款能力、投資價值、詐騙機率或未來營運結果。
        </p>
      </div>
    </section>
  );
}

function DimensionRow({
  explanation,
}: {
  explanation: ScoreDimensionExplanation;
}) {
  const { dimension, details, state } = explanation;
  const scoreText =
    state === "contract_mismatch"
      ? `資料不一致 / ${dimension.max_score}`
      : state === "scored"
        ? `${dimension.score} / ${dimension.max_score}`
        : `資料不足 / ${dimension.max_score}`;
  const hasDetails = details.facts.length > 0 || details.warnings.length > 0;

  return (
    <li className={state === "scored" ? "available" : state}>
      <div className="dimension-label">
        <span>
          {state === "scored" ? (
            <CheckCircle2 aria-hidden="true" size={16} />
          ) : (
            <AlertTriangle aria-hidden="true" size={16} />
          )}
          <strong>{dimension.label}</strong>
        </span>
        <code>{scoreText}</code>
      </div>
      <progress
        aria-label={`${dimension.label}：${scoreText}`}
        aria-valuetext={
          state === "scored"
            ? `${dimension.score} 分，滿分 ${dimension.max_score} 分`
            : state === "unavailable"
              ? "資料不足，未計分"
              : "計分資料不一致"
        }
        max={dimension.max_score}
        value={state === "scored" ? (dimension.score ?? 0) : 0}
      />
      <p className="dimension-summary">{details.summary}</p>
      {hasDetails && (
        <details>
          <summary>查看{dimension.label}的完整計分依據</summary>
          {details.facts.length > 0 && (
            <ul className="evidence-list friendly-evidence-list">
              {details.facts.map((fact) => (
                <li key={fact}>
                  <span>{fact}</span>
                </li>
              ))}
            </ul>
          )}
          {details.warnings.length > 0 && (
            <ul className="dimension-warning-list">
              {details.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          )}
        </details>
      )}
    </li>
  );
}

function PeerMetric({
  label,
  value,
  companyValue,
  medianValue,
  composite = false,
}: {
  label: string;
  value: number | null;
  companyValue: string;
  medianValue: string;
  composite?: boolean;
}) {
  const roundedValue = value === null ? null : Math.round(value * 10) / 10;
  return (
    <article className="peer-metric">
      <div className="peer-metric-title">
        <strong>{label}</strong>
        <span>{roundedValue === null ? "資料不足" : `${composite ? "指標" : "PR"} ${roundedValue}`}</span>
      </div>
      <progress
        aria-label={`${label} ${roundedValue === null ? "資料不足" : `${composite ? "指標" : "PR"} ${roundedValue}`}`}
        max={100}
        value={roundedValue ?? 0}
      />
      <div className="peer-axis" aria-hidden="true">
        <span>{composite ? "0" : "PR 0"}</span>
        <span>{composite ? "刻度中點 50" : "同業中間位置 PR 50"}</span>
        <span>{composite ? "100" : "PR 100"}</span>
      </div>
      <p>
        {roundedValue === null
          ? "目前無法計算相對位置。"
          : composite
            ? "這是年資 PR 與資本 PR 的平均值，不代表高於同業的百分比，也不是公司優劣排名。"
            : "數值越接近 100，這項公開指標在同業樣本中的位置越高；相同數值採中間名次。"}
      </p>
      <dl>
        <div>
          <dt>本公司</dt>
          <dd>{companyValue.replace(/^公司\s*/, "")}</dd>
        </div>
        <div>
          <dt>比較基準</dt>
          <dd>{medianValue.replace(/^同業中位數\s*/, "")}</dd>
        </div>
      </dl>
    </article>
  );
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat("zh-TW", { maximumFractionDigits: 0 }).format(value);
}

function formatDecimal(value: number): string {
  return new Intl.NumberFormat("zh-TW", { maximumFractionDigits: 1 }).format(value);
}

function formatCurrency(value: number | null): string {
  if (value === null) return "資料未提供";
  return new Intl.NumberFormat("zh-TW", {
    style: "currency",
    currency: "TWD",
    maximumFractionDigits: 0,
  }).format(value);
}

function formatDate(value: string): string {
  return value.replaceAll("-", "/");
}

function formatCoveragePercent(coverage: number): string {
  const percent = coverage * 100;
  return new Intl.NumberFormat("zh-TW", {
    maximumFractionDigits: Number.isInteger(percent) ? 0 : 1,
  }).format(percent);
}
