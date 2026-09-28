import { AlertTriangle } from "lucide-react";

import type { ResponseMeta } from "./types";


export interface SnapshotNoticeSource {
  label?: string;
  meta: ResponseMeta;
}

export default function StaleDataNotice({
  sources,
}: {
  sources: SnapshotNoticeSource[];
}) {
  const staleSources = sources.filter(
    ({ meta }) => (meta.data_freshness ?? "live") === "stale_cache",
  );
  if (staleSources.length === 0) return null;

  return (
    <div
      className="stale-data-notice"
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      <AlertTriangle aria-hidden="true" size={21} />
      <div>
        <strong>即時資料暫不可用</strong>
        {staleSources.length === 1 ? (
          <p>
            目前顯示最近一次成功取得的資料，快照時間為{" "}
            <time dateTime={staleSources[0].meta.fetched_at}>
              {formatSnapshotTime(staleSources[0].meta.fetched_at)}
            </time>
            。GCIS 恢復後，重新查詢即可取得即時資料。
          </p>
        ) : (
          <>
            <p>以下公司目前顯示最近一次成功資料，請依各自快照時間判讀：</p>
            <ul>
              {staleSources.map(({ label, meta }, index) => (
                <li key={`${label ?? "snapshot"}-${meta.fetched_at}-${index}`}>
                  {label && <span>{label}：</span>}
                  <time dateTime={meta.fetched_at}>
                    {formatSnapshotTime(meta.fetched_at)}
                  </time>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}


function formatSnapshotTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-TW", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "Asia/Taipei",
  }).format(date);
}
