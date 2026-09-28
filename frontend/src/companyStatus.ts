import type { CompanyStatus } from "./types";

// GCIS company table 7; registration labels are not operating claims.
export const COMPANY_STATUS_DESCRIPTIONS: Record<string, string> = {
  "01": "核准設立", "02": "核准設立，但已命令解散", "03": "重整",
  "04": "解散", "05": "撤銷", "06": "破產", "07": "合併解散",
  "08": "撤回登記", "09": "廢止", "10": "廢止認許", "11": "解散已清算完結",
  "12": "撤銷已清算完結", "13": "廢止已清算完結", "14": "撤回登記已清算完結",
  "15": "撤銷登記已清算完結", "16": "廢止登記已清算完結", "17": "撤銷登記",
  "18": "分割解散", "19": "終止破產", "20": "中止破產", "21": "塗銷破產",
  "22": "破產程序終結(終止)", "23": "破產程序終結(終止)清算中",
  "24": "破產已清算完結", "25": "接管", "26": "撤銷無需清算",
  "27": "撤銷許可", "28": "廢止許可", "29": "撤銷許可已清算完結",
  "30": "廢止許可已清算完結", "31": "清理", "32": "撤銷公司設立", "33": "清理完結",
};

const normalize = (text: string) => text.normalize("NFKC").replace(/[^\p{L}\p{N}]/gu, "");

export function hasStatusConflict(status: CompanyStatus): boolean {
  if (status.description === "來源登記狀態不一致，待確認") return true;
  const official = COMPANY_STATUS_DESCRIPTIONS[status.code ?? ""];
  return !!(official && status.description?.trim() && normalize(official) !== normalize(status.description));
}

export function noScoreLabel(status: CompanyStatus): string {
  if (hasStatusConflict(status)) return "狀態資料待確認，暫不分級";
  if (!status.code || !COMPANY_STATUS_DESCRIPTIONS[status.code]) return "登記狀態待確認，暫不分級";
  if (status.code !== "01") return "依登記狀態不計總分";
  return "資料不足，暫不分級";
}
