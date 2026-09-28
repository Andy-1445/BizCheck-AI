export function formatCompanyAge(
  establishedAt: string | null,
  asOf: string | null | undefined,
  fallbackYears: number | null,
): string {
  const establishedDate = parseDateParts(establishedAt);
  const referenceDate = parseDateParts(asOf);

  if (establishedDate && referenceDate) {
    let totalMonths =
      (referenceDate.year - establishedDate.year) * 12 +
      (referenceDate.month - establishedDate.month);
    if (referenceDate.day < establishedDate.day) totalMonths -= 1;

    if (totalMonths >= 0) return formatYearsAndMonths(totalMonths);
  }

  if (
    fallbackYears !== null &&
    Number.isFinite(fallbackYears) &&
    fallbackYears >= 0
  ) {
    return formatYearsAndMonths(Math.round(fallbackYears * 12));
  }

  return "資料未提供";
}

interface DateParts {
  year: number;
  month: number;
  day: number;
}

function parseDateParts(value: string | null | undefined): DateParts | null {
  if (!value) return null;
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(value);
  if (!match) return null;

  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const date = new Date(Date.UTC(year, month - 1, day));
  if (
    date.getUTCFullYear() !== year ||
    date.getUTCMonth() !== month - 1 ||
    date.getUTCDate() !== day
  ) {
    return null;
  }
  return { year, month, day };
}

function formatYearsAndMonths(totalMonths: number): string {
  const years = Math.floor(totalMonths / 12);
  const months = totalMonths % 12;
  return `${years} 年 ${months} 個月`;
}
