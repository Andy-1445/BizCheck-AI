import {
  buildFriendlyDimensionDetails,
  type FriendlyDimensionDetails,
} from "./friendlyEvidence";
import type {
  BizScoreDimension,
  BizScoreResult,
  CompanyBizScoreResponse,
} from "./types";
import { hasStatusConflict } from "./companyStatus";

export type ScoreExplanationState =
  | "complete"
  | "provisional"
  | "unscored"
  | "contract_mismatch";

export type DimensionExplanationState =
  | "scored"
  | "unavailable"
  | "contract_mismatch";

export interface ScoreDimensionExplanation {
  dimension: BizScoreDimension;
  state: DimensionExplanationState;
  scoreLabel: string;
  details: FriendlyDimensionDetails;
}

export interface BizScoreExplanation {
  version: "1.0";
  generator: "deterministic_bizscore_v1";
  state: ScoreExplanationState;
  stateLabel: string;
  headline: string;
  summary: string;
  composition: string;
  normalizationNote: string | null;
  capNotice: string | null;
  dimensions: ScoreDimensionExplanation[];
  integrityIssues: string[];
}

const COVERAGE_THRESHOLD = 0.8;
const CANONICAL_DIMENSIONS = [
  ["registration_status", "登記狀態", 25],
  ["company_age", "成立年資", 20],
  ["registered_capital_scale", "登記資本規模", 20],
  ["registration_change_recency", "登記異動距今", 15],
  ["peer_relative_position", "同業相對位置", 20],
] as const;

const TOTAL_ELIGIBLE_STATUS_CODES = new Set(["01"]);
const REVIEW_REQUIRED_STATUS_CODES = new Set(["02", "03", "23", "25", "31"]);
const NON_CURRENT_STATUS_CODES = new Set([
  "04",
  "05",
  "06",
  "07",
  "08",
  "09",
  "10",
  "11",
  "12",
  "13",
  "14",
  "15",
  "16",
  "17",
  "18",
  "19",
  "20",
  "21",
  "22",
  "24",
  "26",
  "27",
  "28",
  "29",
  "30",
  "32",
  "33",
]);

export function buildBizScoreExplanation(
  response: CompanyBizScoreResponse,
): BizScoreExplanation {
  const { company, bizscore } = response.data;
  const integrityIssues = validateExplanationContract(bizscore);
  const dimensions = orderDimensions(bizscore.dimensions).map((dimension) => {
    const state = dimensionState(dimension);
    return {
      dimension,
      state,
      scoreLabel: dimensionScoreLabel(dimension, state),
      details: buildFriendlyDimensionDetails(dimension, {
        establishedAt: company.established_at,
        asOf: bizscore.as_of,
      }),
    } satisfies ScoreDimensionExplanation;
  });

  const state = explanationState(bizscore, integrityIssues);
  const coverageLabel = formatCoverage(bizscore.coverage);
  const missingLabels = dimensions
    .filter((item) => item.state === "unavailable")
    .map((item) => item.dimension.label);

  return {
    version: "1.0",
    generator: "deterministic_bizscore_v1",
    state,
    stateLabel: explanationStateLabel(state),
    headline: explanationHeadline(state, bizscore.score),
    summary: explanationSummary({
      state,
      bizscore,
      coverageLabel,
      missingLabels,
      statusCode: company.status.code,
      statusDescription: company.status.description,
      sourcePartial: response.meta.partial,
      sourceWarnings: response.meta.warnings,
      integrityIssues,
    }),
    composition: dimensionComposition(dimensions),
    normalizationNote:
      state === "provisional" && bizscore.coverage < 1
        ? "部分構面未計分時，BizScore v1 會依可用構面的權重換算暫定總分；下列構面原始分數不能直接相加後取代畫面總分。"
        : null,
    capNotice:
      bizscore.status_cap !== null && bizscore.status_cap < 100
        ? `依目前官方登記狀態，BizScore v1 將總分上限設為 ${bizscore.status_cap} 分；各構面仍逐項呈現。`
        : null,
    dimensions,
    integrityIssues,
  };
}

function explanationState(
  bizscore: BizScoreResult,
  integrityIssues: string[],
): ScoreExplanationState {
  if (integrityIssues.length > 0) return "contract_mismatch";
  if (bizscore.score === null) return "unscored";
  return bizscore.provisional ? "provisional" : "complete";
}

function explanationStateLabel(state: ScoreExplanationState): string {
  switch (state) {
    case "complete":
      return "完整分數";
    case "provisional":
      return "暫定分數";
    case "unscored":
      return "未顯示總分";
    case "contract_mismatch":
      return "資料待確認";
  }
}

function explanationHeadline(
  state: ScoreExplanationState,
  score: number | null,
): string {
  switch (state) {
    case "complete":
      return `為什麼是 ${score} 分？`;
    case "provisional":
      return `為什麼目前顯示暫定 ${score} 分？`;
    case "unscored":
      return "為什麼目前沒有數字總分？";
    case "contract_mismatch":
      return "為什麼目前無法可靠說明這個分數？";
  }
}

function explanationSummary({
  state,
  bizscore,
  coverageLabel,
  missingLabels,
  statusCode,
  statusDescription,
  sourcePartial,
  sourceWarnings,
  integrityIssues,
}: {
  state: ScoreExplanationState;
  bizscore: BizScoreResult;
  coverageLabel: string;
  missingLabels: string[];
  statusCode: string | null;
  statusDescription: string | null;
  sourcePartial: boolean;
  sourceWarnings: string[];
  integrityIssues: string[];
}): string {
  if (state === "contract_mismatch") {
    return `本次計分資料有 ${integrityIssues.length} 項彼此不一致，為避免產生錯誤因果，系統保留 API 回傳值，但不推測分數成因。請稍後重試或查看資料限制。`;
  }

  if (state === "complete") {
    return `五個構面都有可用資料，BizScore v1 固定規則回傳 ${bizscore.score} 分。下方逐項顯示各構面分數與對應的公開資料理由。`;
  }

  if (state === "provisional") {
    const reasons: string[] = [];
    if (missingLabels.length > 0) {
      reasons.push(`${quoteList(missingLabels)}目前未計分`);
    }
    if (sourcePartial) reasons.push("來源資料被標記為部分資料");
    if (sourceWarnings.length > 0) reasons.push("來源另有資料提醒");
    if (reasons.length === 0 && bizscore.warnings.length > 0) {
      reasons.push("本次計分另有資料限制提醒");
    }
    const reasonSentence =
      reasons.length > 0
        ? `${joinNatural(reasons)}。`
        : "系統已將本次結果標記為暫定，請連同資料限制判讀。";
    return `BizScore v1 固定規則回傳暫定 ${bizscore.score} 分，可計分資料覆蓋率為 ${coverageLabel}。${reasonSentence}這個數字只反映目前可用資料；未計分不等於 0 分。`;
  }

  const statusReason = noScoreStatusReason(statusCode, statusDescription);
  const missingSentence =
    missingLabels.length > 0
      ? `${quoteList(missingLabels)}目前資料不足。`
      : "";
  if (statusReason) {
    return `${statusReason}可計分資料覆蓋率為 ${coverageLabel}。${missingSentence}沒有數字總分不代表 0 分。`;
  }
  if (bizscore.coverage < COVERAGE_THRESHOLD) {
    return `可計分資料覆蓋率為 ${coverageLabel}，低於 BizScore v1 的 80% 顯示門檻。${missingSentence}因此目前不顯示數字總分；這不代表 0 分。`;
  }
  return `API 目前沒有回傳數字總分；可計分資料覆蓋率為 ${coverageLabel}。${missingSentence}為避免猜測原因，本區只呈現既有構面與資料限制。`;
}

function noScoreStatusReason(
  statusCode: string | null,
  statusDescription: string | null,
): string | null {
  const description = statusDescription ? `「${statusDescription}」` : "目前狀態";
  if (hasStatusConflict({ code: statusCode, description: statusDescription })) {
    return "來源登記狀態彼此不一致，尚無法確認適用的計分規則，因此暫不計算登記狀態分數與數字總分；請重新查詢或核對官方資料。";
  }
  if (statusCode !== null && REVIEW_REQUIRED_STATUS_CODES.has(statusCode)) {
    return `官方登記狀態為${description}，需要優先進一步查核；BizScore v1 因此不產生數字總分。`;
  }
  if (statusCode !== null && NON_CURRENT_STATUS_CODES.has(statusCode)) {
    return `官方登記狀態為${description}，不符合 BizScore v1 僅對核准設立公司產生總分的規則，因此不產生數字總分；這不是對實際營運或履約能力的判斷。`;
  }
  if (statusCode === null || !TOTAL_ELIGIBLE_STATUS_CODES.has(statusCode)) {
    return "目前登記狀態無法由 BizScore v1 辨識，因此暫不產生數字總分。";
  }
  return null;
}

function dimensionComposition(
  dimensions: ScoreDimensionExplanation[],
): string {
  const parts = dimensions.map((item) => {
    if (item.state === "contract_mismatch") {
      return `${item.dimension.label}的計分資料不一致`;
    }
    if (item.state === "unavailable") {
      return `${item.dimension.label}資料不足`;
    }
    return `${item.dimension.label} ${item.scoreLabel}`;
  });
  return `五個構面的原始結果為：${parts.join("、")}。`;
}

function dimensionState(
  dimension: BizScoreDimension,
): DimensionExplanationState {
  if (dimension.available !== (dimension.score !== null)) {
    return "contract_mismatch";
  }
  return dimension.available ? "scored" : "unavailable";
}

function dimensionScoreLabel(
  dimension: BizScoreDimension,
  state: DimensionExplanationState,
): string {
  if (state === "contract_mismatch") return "計分資料不一致";
  if (state === "unavailable") return `資料不足／${dimension.max_score} 分`;
  return `${dimension.score}／${dimension.max_score} 分`;
}

function orderDimensions(dimensions: BizScoreDimension[]): BizScoreDimension[] {
  const order = new Map<string, number>(
    CANONICAL_DIMENSIONS.map(([key], index) => [key, index]),
  );
  return dimensions
    .map((dimension, index) => ({ dimension, index }))
    .sort((left, right) => {
      const leftOrder = order.get(left.dimension.key) ?? Number.MAX_SAFE_INTEGER;
      const rightOrder = order.get(right.dimension.key) ?? Number.MAX_SAFE_INTEGER;
      return leftOrder === rightOrder ? left.index - right.index : leftOrder - rightOrder;
    })
    .map(({ dimension }) => dimension);
}

function validateExplanationContract(bizscore: BizScoreResult): string[] {
  const issues: string[] = [];
  const counts = new Map<string, number>();
  for (const dimension of bizscore.dimensions) {
    counts.set(dimension.key, (counts.get(dimension.key) ?? 0) + 1);
    if (dimension.available !== (dimension.score !== null)) {
      issues.push(`${dimension.key}: available 與 score 不一致`);
    }
    if (
      dimension.score !== null &&
      (dimension.score < 0 || dimension.score > dimension.max_score)
    ) {
      issues.push(`${dimension.key}: score 超出構面範圍`);
    }
  }

  for (const [key, , maxScore] of CANONICAL_DIMENSIONS) {
    if ((counts.get(key) ?? 0) !== 1) {
      issues.push(`${key}: 構面數量不是 1`);
      continue;
    }
    const dimension = bizscore.dimensions.find((item) => item.key === key);
    if (dimension && dimension.max_score !== maxScore) {
      issues.push(`${key}: max_score 不符合 BizScore v1`);
    }
  }

  for (const key of counts.keys()) {
    if (!CANONICAL_DIMENSIONS.some(([canonicalKey]) => canonicalKey === key)) {
      issues.push(`${key}: 未知構面`);
    }
  }

  const derivedMissing = new Set(
    bizscore.dimensions
      .filter((dimension) => !dimension.available || dimension.score === null)
      .map((dimension) => dimension.key),
  );
  const reportedMissing = new Set(bizscore.missing_dimensions);
  if (!sameStringSet(derivedMissing, reportedMissing)) {
    issues.push("missing_dimensions 與構面狀態不一致");
  }

  const peerDimension = bizscore.dimensions.find(
    (dimension) => dimension.key === "peer_relative_position",
  );
  const peerCopy = bizscore.peer_benchmark.dimension;
  if (
    peerDimension &&
    (peerCopy.key !== peerDimension.key ||
      peerCopy.available !== peerDimension.available ||
      peerCopy.score !== peerDimension.score ||
      peerCopy.max_score !== peerDimension.max_score)
  ) {
    issues.push("peer_benchmark.dimension 與五構面結果不一致");
  }

  if ((bizscore.score === null) !== (bizscore.band === null)) {
    issues.push("score 與 band 的空值狀態不一致");
  }
  if (bizscore.score === null && bizscore.provisional) {
    issues.push("無數字總分不可標示 provisional");
  }
  if (bizscore.score !== null && bizscore.coverage < COVERAGE_THRESHOLD) {
    issues.push("覆蓋率低於 80% 卻有數字總分");
  }
  if (bizscore.score !== null && !bizscore.provisional && bizscore.coverage < 1) {
    issues.push("部分覆蓋的數字總分未標示 provisional");
  }
  if (
    bizscore.score !== null &&
    bizscore.status_cap !== null &&
    bizscore.score > bizscore.status_cap
  ) {
    issues.push("總分高於登記狀態上限");
  }
  return [...new Set(issues)];
}

function sameStringSet(left: Set<string>, right: Set<string>): boolean {
  if (left.size !== right.size) return false;
  for (const value of left) {
    if (!right.has(value)) return false;
  }
  return true;
}

function formatCoverage(coverage: number): string {
  const percent = coverage * 100;
  return `${new Intl.NumberFormat("zh-TW", {
    maximumFractionDigits: Number.isInteger(percent) ? 0 : 1,
  }).format(percent)}%`;
}

function quoteList(values: string[]): string {
  return values.map((value) => `「${value}」`).join("、");
}

function joinNatural(values: string[]): string {
  if (values.length <= 1) return values[0] ?? "";
  return `${values.slice(0, -1).join("、")}，且${values.at(-1)}`;
}
