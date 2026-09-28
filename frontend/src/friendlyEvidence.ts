import { formatCompanyAge } from "./formatters";
import type { BizScoreDimension } from "./types";

export interface FriendlyDimensionDetails {
  summary: string;
  facts: string[];
  warnings: string[];
}

export interface FriendlyEvidenceContext {
  establishedAt?: string | null;
  asOf?: string | null;
}

const INDUSTRY_NAMES: Record<string, string> = {
  A: "農、林、漁、牧業",
  B: "礦業及土石採取業",
  C: "製造業",
  D: "水電燃氣業",
  E: "營造及工程業",
  F: "零售、批發及餐飲業",
  G: "運輸、倉儲及通信業",
  H: "金融、保險及不動產業",
  I: "專業、科學及技術服務業",
  J: "文化、運動、休閒及其他服務業",
  Z: "其他未分類業",
};

const AGE_RULES: Record<string, string> = {
  "age>=10": "成立年資 10 年以上",
  "5<=age<10": "成立年資滿 5 年、未滿 10 年",
  "3<=age<5": "成立年資滿 3 年、未滿 5 年",
  "1<=age<3": "成立年資滿 1 年、未滿 3 年",
  "0<=age<1": "成立年資未滿 1 年",
};

const CAPITAL_RULES: Record<string, string> = {
  "capital>=20000000": "登記資本額達新臺幣 2,000 萬元以上",
  "5000000<=capital<20000000":
    "登記資本額達新臺幣 500 萬元、未滿 2,000 萬元",
  "1000000<=capital<5000000":
    "登記資本額達新臺幣 100 萬元、未滿 500 萬元",
  "500000<=capital<1000000":
    "登記資本額達新臺幣 50 萬元、未滿 100 萬元",
  "0<capital<500000": "登記資本額高於 0 元、未滿新臺幣 50 萬元",
  "capital=0": "登記資本額為 0 元",
};

const CHANGE_RULES: Record<string, string> = {
  "days>=1825": "最近一次登記異動距今已滿 5 年",
  "1095<=days<1825": "最近一次登記異動距今已滿 3 年、未滿 5 年",
  "365<=days<1095": "最近一次登記異動距今已滿 1 年、未滿 3 年",
  "90<=days<365": "最近一次登記異動距今已滿 90 天、未滿 1 年",
  "0<=days<90": "最近一次登記異動距今未滿 90 天",
};

const PEER_RULES: Record<string, string> = {
  "peer_index>=80": "年資與資本 PR 的平均值達 80 以上",
  "60<=peer_index<80": "年資與資本 PR 的平均值介於 60 至未滿 80",
  "40<=peer_index<60": "年資與資本 PR 的平均值介於 40 至未滿 60",
  "20<=peer_index<40": "年資與資本 PR 的平均值介於 20 至未滿 40",
  "0<=peer_index<20": "年資與資本 PR 的平均值低於 20",
};

const STATUS_RULES: Record<string, string> = {
  status_01: "目前為核准設立狀態",
  status_02: "公司核准設立，但已命令解散",
  status_03: "公司目前為重整狀態",
  status_04: "公司目前為解散狀態",
  status_05: "公司目前為撤銷狀態",
  status_06: "公司目前為破產狀態",
  status_07: "公司目前為合併解散狀態",
  status_08: "公司目前為撤回登記狀態",
  status_09: "公司目前為廢止狀態",
  status_10: "公司目前為廢止認許狀態",
  status_11: "公司目前為解散已清算完結狀態",
  status_12: "公司目前為撤銷已清算完結狀態",
  status_13: "公司目前為廢止已清算完結狀態",
  status_14: "公司目前為撤回登記已清算完結狀態",
  status_15: "公司目前為撤銷登記已清算完結狀態",
  status_16: "公司目前為廢止登記已清算完結狀態",
  status_17: "公司目前為撤銷登記狀態",
  status_18: "公司目前為分割解散狀態",
  status_19: "公司目前為終止破產狀態",
  status_20: "公司目前為中止破產狀態",
  status_21: "公司目前為塗銷破產狀態",
  status_22: "公司目前為破產程序終結（終止）狀態",
  status_23: "公司目前為破產程序終結（終止）清算中狀態",
  status_24: "公司目前為破產已清算完結狀態",
  status_25: "公司目前為接管狀態",
  status_26: "公司目前為撤銷無需清算狀態",
  status_27: "公司目前為撤銷許可狀態",
  status_28: "公司目前為廢止許可狀態",
  status_29: "公司目前為撤銷許可已清算完結狀態",
  status_30: "公司目前為廢止許可已清算完結狀態",
  status_31: "公司目前為清理狀態",
  status_32: "公司目前為撤銷公司設立狀態",
  status_33: "公司目前為清理完結狀態",
};

export function friendlyBizScoreRule(rule: string): string | null {
  return (
    STATUS_RULES[rule] ??
    AGE_RULES[rule] ??
    CAPITAL_RULES[rule] ??
    CHANGE_RULES[rule] ??
    PEER_RULES[rule] ??
    null
  );
}

export function buildFriendlyDimensionDetails(
  dimension: BizScoreDimension,
  context: FriendlyEvidenceContext = {},
): FriendlyDimensionDetails {
  const evidence = toEvidenceMap(dimension.evidence);
  let facts: string[];

  switch (dimension.key) {
    case "registration_status":
      facts = registrationStatusFacts(dimension, evidence);
      break;
    case "company_age":
      facts = companyAgeFacts(dimension, evidence, context);
      break;
    case "registered_capital_scale":
      facts = registeredCapitalFacts(dimension, evidence);
      break;
    case "registration_change_recency":
      facts = registrationChangeFacts(dimension, evidence);
      break;
    case "peer_relative_position":
      facts = peerPositionFacts(dimension, evidence);
      break;
    default:
      facts = dimension.available
        ? [`本構面得 ${scoreText(dimension)}。`]
        : [];
  }

  const warnings = friendlyWarnings(dimension);
  const summary = buildDimensionSummary(dimension, evidence, warnings);

  return {
    summary,
    facts: unique(facts).filter((fact) => fact !== summary),
    warnings,
  };
}

function buildDimensionSummary(
  dimension: BizScoreDimension,
  evidence: Map<string, string>,
  warnings: string[],
): string {
  if (!dimension.available || dimension.score === null) {
    const reason = warnings[0] ?? `${dimension.label}目前沒有可用資料，暫不計分。`;
    return `${stripFinalPunctuation(reason)}；未計分不等於 0 分。`;
  }

  const ruleLabel = friendlyBizScoreRule(evidence.get("rule") ?? "");
  if (ruleLabel) {
    return `${ruleLabel}，本構面得 ${scoreText(dimension)}。`;
  }

  return `依已取得的公開資料，本構面得 ${scoreText(dimension)}。`;
}

function registrationStatusFacts(
  dimension: BizScoreDimension,
  evidence: Map<string, string>,
): string[] {
  const code = evidence.get("status.code");
  const description =
    evidence.get("official_status") ?? evidence.get("status.description");
  const facts: string[] = [];

  if (description) {
    facts.push(
      code
        ? `目前官方登記狀態：${description}（代碼 ${code}）。`
        : `目前官方登記狀態：${description}。`,
    );
  }
  if (dimension.available) {
    facts.push(`依目前登記狀態，本構面得 ${scoreText(dimension)}。`);
  }
  return facts;
}

function companyAgeFacts(
  dimension: BizScoreDimension,
  evidence: Map<string, string>,
  context: FriendlyEvidenceContext,
): string[] {
  const age = finiteNumber(evidence.get("company_age_years"));
  const ruleLabel = friendlyBizScoreRule(evidence.get("rule") ?? "");
  const facts: string[] = [];

  if (age !== null) {
    facts.push(
      `截至計分基準日，公司成立年資：${formatCompanyAge(
        context.establishedAt ?? null,
        context.asOf ?? null,
        age,
      )}。`,
    );
  }
  if (ruleLabel && dimension.available) {
    facts.push(`${ruleLabel}，本構面得 ${scoreText(dimension)}。`);
  }
  return facts;
}

function registeredCapitalFacts(
  dimension: BizScoreDimension,
  evidence: Map<string, string>,
): string[] {
  const capital = finiteNumber(evidence.get("capital.registered"));
  const ruleLabel = friendlyBizScoreRule(evidence.get("rule") ?? "");
  const facts: string[] = [];

  if (capital !== null) {
    facts.push(`登記資本額：新臺幣 ${formatInteger(capital)} 元。`);
  }
  if (ruleLabel && dimension.available) {
    facts.push(`${ruleLabel}，本構面得 ${scoreText(dimension)}。`);
  }
  return facts;
}

function registrationChangeFacts(
  dimension: BizScoreDimension,
  evidence: Map<string, string>,
): string[] {
  const lastChangedAt = evidence.get("last_changed_at");
  const establishedAt = evidence.get("established_at");
  const asOf = evidence.get("as_of");
  const days = finiteNumber(evidence.get("days_since_last_change"));
  const ruleLabel = friendlyBizScoreRule(evidence.get("rule") ?? "");
  const facts: string[] = [];

  if (lastChangedAt) facts.push(`最後核准異動日：${formatDate(lastChangedAt)}。`);
  if (establishedAt) facts.push(`公司設立日：${formatDate(establishedAt)}。`);
  if (asOf) facts.push(`計分基準日：${formatDate(asOf)}。`);
  if (days !== null) facts.push(`距計分基準日 ${formatInteger(days)} 天。`);
  if (ruleLabel && dimension.available) {
    facts.push(`${ruleLabel}，本構面得 ${scoreText(dimension)}。`);
  }
  return facts;
}

function peerPositionFacts(
  dimension: BizScoreDimension,
  evidence: Map<string, string>,
): string[] {
  const code = evidence.get("industry.code");
  const benchmarkVersion = evidence.get("benchmark.version");
  const sampleSize = finiteNumber(evidence.get("peer.sample_size"));
  const ageMedian = finiteNumber(evidence.get("peer.company_age_median"));
  const capitalMedian = finiteNumber(
    evidence.get("peer.registered_capital_median"),
  );
  const agePr = finiteNumber(evidence.get("age_pr"));
  const capitalPr = finiteNumber(evidence.get("capital_pr"));
  const peerIndex = finiteNumber(evidence.get("peer_index"));
  const ruleLabel = friendlyBizScoreRule(evidence.get("rule") ?? "");
  const facts: string[] = [];

  if (code) {
    const industryName = INDUSTRY_NAMES[code];
    facts.push(
      industryName
        ? `比較產業：${code} 類－${industryName}。`
        : `比較產業：${code} 類。`,
    );
  }
  const benchmarkDate = benchmarkVersion?.match(
    /^benchmark-(\d{4})-(\d{2})-(\d{2})-/,
  );
  if (benchmarkDate) {
    facts.push(
      `同業資料基準日：${benchmarkDate[1]}/${benchmarkDate[2]}/${benchmarkDate[3]}。`,
    );
  }
  if (sampleSize !== null) {
    facts.push(`有效同業樣本：${formatInteger(sampleSize)} 家。`);
  }
  if (ageMedian !== null) {
    facts.push(`同業成立年資中位數：${formatDecimal(ageMedian)} 年。`);
  }
  if (capitalMedian !== null) {
    facts.push(
      `同業登記資本額中位數：新臺幣 ${formatInteger(capitalMedian)} 元。`,
    );
  }
  if (agePr !== null) {
    facts.push(`成立年資的同業相對位置為 PR ${formatDecimal(agePr)}；相同年資的樣本採中間名次。`);
  }
  if (capitalPr !== null) {
    facts.push(`登記資本的同業相對位置為 PR ${formatDecimal(capitalPr)}；相同資本的樣本採中間名次。`);
  }
  if (peerIndex !== null) {
    facts.push(`年資 PR 與資本 PR 各占一半，綜合指標為 ${formatDecimal(peerIndex)}；這不是另一個百分位排名。`);
  }
  if (ruleLabel && dimension.available) {
    facts.push(`${ruleLabel}，本構面得 ${scoreText(dimension)}。`);
  }
  return facts;
}

function friendlyWarnings(dimension: BizScoreDimension): string[] {
  const warnings = dimension.warnings.map((warning) => {
    const capMatch = warning.match(/capped at (\d+)/i);
    if (capMatch) return `因目前登記狀態，BizScore 總分上限為 ${capMatch[1]} 分。`;

    const excludedMatch = warning.match(/^(\d+) incomplete benchmark sample/i);
    if (excludedMatch) {
      return `同業比較已排除 ${formatInteger(Number(excludedMatch[1]))} 筆資料不完整的樣本。`;
    }

    if (warning.includes("GCIS_STATUS_CONFLICT")) {
      return "來源登記狀態彼此不一致，尚待核對，本構面與總分暫不計分。";
    }
    if (warning.includes("non-current")) {
      return "依目前登記狀態，BizScore 不產生總分；此規則不判斷公司的實際營運或履約能力。";
    }
    if (warning.includes("enhanced review")) {
      return "目前登記狀態需要優先進一步查核，因此暫不計算總分。";
    }
    if (warning.includes("status.code")) {
      return "公司登記狀態無法辨識，本構面暫不計分。";
    }

    switch (dimension.key) {
      case "company_age":
        return "成立年資資料缺漏或格式不符，本構面暫不計分。";
      case "registered_capital_scale":
        return "登記資本額資料缺漏或格式不符，本構面暫不計分。";
      case "registration_change_recency":
        if (warning.includes("later than as_of")) {
          return "登記日期晚於計分基準日，無法計算異動距今時間。";
        }
        if (warning.includes("earlier than established_at")) {
          return "最後異動日早於公司設立日，日期順序異常，本構面暫不計分。";
        }
        return "最後異動日期缺漏或格式不符，本構面暫不計分。";
      case "peer_relative_position":
        return "目前沒有足夠的有效同業資料，本構面暫不計分。";
      case "registration_status":
        return "公司登記狀態資料缺漏，本構面暫不計分。";
      default:
        return "部分來源資料不符合本構面計分規則，因此未納入計分。";
    }
  });

  return unique(warnings);
}

function toEvidenceMap(values: string[]): Map<string, string> {
  const result = new Map<string, string>();
  for (const value of values) {
    const separator = value.indexOf("=");
    if (separator <= 0) continue;
    result.set(value.slice(0, separator), value.slice(separator + 1));
  }
  return result;
}

function scoreText(dimension: BizScoreDimension): string {
  return dimension.score === null
    ? `資料不足／${dimension.max_score} 分`
    : `${dimension.score}／${dimension.max_score} 分`;
}

function finiteNumber(value: string | undefined): number | null {
  if (value === undefined || value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat("zh-TW", { maximumFractionDigits: 0 }).format(
    value,
  );
}

function formatDecimal(value: number): string {
  return new Intl.NumberFormat("zh-TW", { maximumFractionDigits: 1 }).format(
    value,
  );
}

function formatDate(value: string): string {
  return value.replaceAll("-", "/");
}

function unique(values: string[]): string[] {
  return [...new Set(values)];
}

function stripFinalPunctuation(value: string): string {
  return value.replace(/[。；;，,\s]+$/u, "");
}
