import type {
  ApiErrorDetail,
  CompanyComparisonFallback,
  CompanyComparisonLLMOutput,
  CompanyComparisonAnalysisResponse,
  CompanyAnalysisLLMInput,
  CompanyAnalysisResponse,
  CompanyData,
  CompanyBizScoreResponse,
  CompanyComparisonResponse,
  CompanyResponse,
  CompanySearchResponse,
} from "./types";
import { COMPANY_COMPARISON_DISCLAIMER } from "./types";
import { COMPANY_COMPARISON_AI_DISCLAIMER } from "./types";
import { hasStatusConflict } from "./companyStatus";

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";
const API_BASE_URL = configuredBaseUrl.replace(/\/$/, "");
const BIZSCORE_V1_DIMENSIONS = [
  ["registration_status", 25],
  ["company_age", 20],
  ["registered_capital_scale", 20],
  ["registration_change_recency", 15],
  ["peer_relative_position", 20],
] as const;
const BIZSCORE_COVERAGE_EPSILON = 1e-9;

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryable: boolean;

  constructor(status: number, detail: ApiErrorDetail) {
    super(detail.message);
    this.name = "ApiError";
    this.status = status;
    this.code = detail.code;
    this.retryable = detail.retryable;
  }
}

export async function getCompany(
  taxId: string,
  signal?: AbortSignal,
): Promise<CompanyResponse> {
  return requestJson<CompanyResponse>(
    `${API_BASE_URL}/companies/${encodeURIComponent(taxId)}`,
    signal,
  );
}

export async function getCompanyBizScore(
  taxId: string,
  signal?: AbortSignal,
): Promise<CompanyBizScoreResponse> {
  const response = await requestJson<unknown>(
    `${API_BASE_URL}/companies/${encodeURIComponent(taxId)}/bizscore`,
    signal,
  );
  if (!isCompanyBizScoreResponse(response, taxId)) {
    throw new ApiError(502, {
      code: "BIZSCORE_INVALID_RESPONSE",
      message:
        "BizScore 回應格式或計分欄位彼此不一致，因此未顯示這次公司資料與分數。請重新查詢後再試。",
      retryable: true,
    });
  }
  return response;
}

export async function compareCompanies(
  taxIds: string[],
  signal?: AbortSignal,
): Promise<CompanyComparisonResponse> {
  const response = await requestJson<unknown>(
    `${API_BASE_URL}/companies/compare`,
    signal,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tax_ids: taxIds }),
    },
  );
  if (!isCompanyComparisonResponse(response, taxIds)) {
    throw new ApiError(502, {
      code: "COMPARISON_INVALID_RESPONSE",
      message: "公司比較回應格式不完整，因此未顯示這次比較結果。",
      retryable: true,
    });
  }
  return response;
}

export async function createCompanyComparisonAnalysis(
  expectedComparison: CompanyComparisonResponse,
  forceRefresh = false,
  signal?: AbortSignal,
): Promise<CompanyComparisonAnalysisResponse> {
  const taxIds = expectedComparison.meta.requested_tax_ids;
  const response = await requestJson<unknown>(
    `${API_BASE_URL}/companies/compare/analysis`,
    signal,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tax_ids: taxIds,
        force_refresh: forceRefresh,
      }),
    },
  );
  if (!isCompanyComparisonAnalysisResponse(response, taxIds)) {
    throw new ApiError(502, {
      code: "AI_COMPARISON_INVALID_RESPONSE",
      message: "AI 查核備忘錄回應格式不完整，因此未顯示這次內容。",
      retryable: true,
    });
  }
  if (!hasSameStableComparison(response.data.comparison, expectedComparison)) {
    throw new ApiError(409, {
      code: "AI_COMPARISON_SNAPSHOT_MISMATCH",
      message: "AI 回應引用的公司資料已和目前比較表不同，請先重新建立上方比較表。",
      retryable: false,
    });
  }
  return response;
}

function hasSameStableComparison(
  received: CompanyComparisonResponse,
  expected: CompanyComparisonResponse,
): boolean {
  return stableComparisonJson(received) === stableComparisonJson(expected);
}

function stableComparisonJson(value: CompanyComparisonResponse): string {
  return stableJson(
    value,
    new Set([
      "generated_at",
      "fetched_at",
      "data_freshness",
      "fallback_reason",
    ]),
  );
}

function stableJson(value: unknown, ignoredKeys: ReadonlySet<string>): string {
  return JSON.stringify(toStableValue(value, ignoredKeys));
}

function toStableValue(
  value: unknown,
  ignoredKeys: ReadonlySet<string>,
): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => toStableValue(item, ignoredKeys));
  }
  if (!isRecord(value)) return value;
  const stable: Record<string, unknown> = {};
  for (const key of Object.keys(value).sort()) {
    if (ignoredKeys.has(key)) continue;
    stable[key] = toStableValue(value[key], ignoredKeys);
  }
  return stable;
}

export async function createCompanyAnalysis(
  expectedSnapshot: CompanyBizScoreResponse,
  signal?: AbortSignal,
): Promise<CompanyAnalysisResponse> {
  const taxId = expectedSnapshot.data.company.tax_id;
  const response = await requestJson<unknown>(
    `${API_BASE_URL}/companies/${encodeURIComponent(taxId)}/analysis`,
    signal,
    { method: "POST" },
  );
  if (!isCompanyAnalysisResponse(response)) {
    throw new ApiError(502, {
      code: "AI_ANALYSIS_INVALID_RESPONSE",
      message: "AI 分析回應格式不完整，因此未顯示這份報告。",
      retryable: true,
    });
  }
  if (
    response.data.analysis.provenance.company_tax_id !== taxId ||
    companyAnalysisInputSnapshotKey(response.data.input) !==
      companyAnalysisSnapshotKey(expectedSnapshot)
  ) {
    throw new ApiError(409, {
      code: "AI_ANALYSIS_SNAPSHOT_MISMATCH",
      message:
        "AI 回應引用的公司資料或 BizScore 已和目前畫面不同，因此未顯示這份報告。請重新查詢這家公司，再產生 AI 分析。",
      retryable: false,
    });
  }
  return response;
}

export function companyAnalysisSnapshotKey(
  response: CompanyBizScoreResponse,
): string {
  return stableJson(
    {
      company: response.data.company,
      bizscore: response.data.bizscore,
      source_meta: response.meta,
    },
    new Set(["fetched_at", "data_freshness", "fallback_reason"]),
  );
}

function companyAnalysisInputSnapshotKey(
  input: CompanyAnalysisLLMInput,
): string {
  return stableJson(
    {
      company: input.company,
      bizscore: input.bizscore,
      source_meta: input.source_meta,
    },
    new Set(["fetched_at", "data_freshness", "fallback_reason"]),
  );
}

export async function searchCompanies(
  keyword: string,
  signal?: AbortSignal,
): Promise<CompanySearchResponse> {
  const params = new URLSearchParams({ q: keyword, status: "01" });
  return requestJson<CompanySearchResponse>(
    `${API_BASE_URL}/companies/search?${params.toString()}`,
    signal,
  );
}

async function requestJson<T>(
  url: string,
  signal?: AbortSignal,
  init?: RequestInit,
): Promise<T> {
  let response: Response;
  try {
    const headers = new Headers(init?.headers);
    if (!headers.has("Accept")) headers.set("Accept", "application/json");
    response = await fetch(url, {
      ...init,
      headers,
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError(0, {
      code: "NETWORK_ERROR",
      message: "無法連線至 BizCheck API，請確認後端服務是否已啟動。",
      retryable: true,
    });
  }

  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(response.status, parseErrorDetail(body, response.status));
  }
  return body as T;
}

function parseErrorDetail(body: unknown, status: number): ApiErrorDetail {
  if (isRecord(body) && isErrorDetail(body.detail)) {
    return body.detail;
  }

  return {
    code: `HTTP_${status}`,
    message: status >= 500 ? "服務暫時無法使用，請稍後再試。" : "查詢內容格式不正確。",
    retryable: status >= 500,
  };
}

function isErrorDetail(value: unknown): value is ApiErrorDetail {
  return (
    isRecord(value) &&
    typeof value.code === "string" &&
    typeof value.message === "string" &&
    typeof value.retryable === "boolean"
  );
}

function isCompanyAnalysisResponse(
  value: unknown,
): value is CompanyAnalysisResponse {
  if (!isRecord(value) || !isRecord(value.data) || !isRecord(value.meta)) {
    return false;
  }
  const { input, analysis } = value.data;
  if (
    !isRecord(input) ||
    !isRecord(input.company) ||
    !isRecord(input.bizscore) ||
    !isRecord(input.source_meta) ||
    !isRecord(analysis) ||
    !isRecord(analysis.provenance)
  ) {
    return false;
  }
  const usage = value.meta.usage;
  return (
    typeof input.company.tax_id === "string" &&
    typeof input.bizscore.as_of === "string" &&
    typeof input.bizscore.provisional === "boolean" &&
    Array.isArray(input.bizscore.dimensions) &&
    typeof input.source_meta.partial === "boolean" &&
    Array.isArray(input.source_meta.warnings) &&
    (analysis.status === "completed" ||
      analysis.status === "insufficient_data") &&
    typeof analysis.headline === "string" &&
    typeof analysis.overall_observation === "string" &&
    Array.isArray(analysis.findings) &&
    analysis.findings.every(isAnalysisFinding) &&
    Array.isArray(analysis.verification_items) &&
    analysis.verification_items.every(isVerificationItem) &&
    Array.isArray(analysis.limitations) &&
    analysis.limitations.every(isAnalysisLimitation) &&
    typeof analysis.disclaimer === "string" &&
    typeof analysis.provenance.company_tax_id === "string" &&
    typeof analysis.provenance.benchmark_catalog_version === "string" &&
    typeof value.meta.provider === "string" &&
    typeof value.meta.model === "string" &&
    typeof value.meta.generated_at === "string" &&
    typeof value.meta.prompt_version === "string" &&
    typeof value.meta.input_schema_version === "string" &&
    typeof value.meta.output_schema_version === "string" &&
    (usage === null ||
      (isRecord(usage) && typeof usage.total_tokens === "number"))
  );
}

function isCompanyComparisonAnalysisResponse(
  value: unknown,
  requestedTaxIds: string[],
): value is CompanyComparisonAnalysisResponse {
  if (!isRecord(value) || !isRecord(value.data) || !isRecord(value.meta)) {
    return false;
  }
  const { data, meta } = value;
  if (
    !isCompanyComparisonResponse(data.comparison, requestedTaxIds) ||
    !isRecord(meta.cache) ||
    !isRecord(meta.fallback) ||
    meta.analysis_version !== "1.0" ||
    !isNonEmptyString(meta.provider) ||
    !isNonEmptyString(meta.model) ||
    !isStringOrNull(meta.provider_response_id) ||
    !isStringOrNull(meta.provider_request_id) ||
    !isStringOrNull(meta.service_tier) ||
    !isOffsetDateTime(meta.generated_at) ||
    meta.prompt_version !== "1.0" ||
    !isSha256(meta.system_prompt_sha256) ||
    meta.input_schema_version !== "1.0" ||
    meta.output_schema_version !== "1.0" ||
    !isSha256(meta.input_sha256) ||
    !(meta.output_sha256 === null || isSha256(meta.output_sha256)) ||
    !isIntegerInRange(meta.attempts, 0, 3) ||
    !isNonNegativeInteger(meta.duration_ms) ||
    !isTokenUsageOrNull(meta.usage) ||
    !isComparisonCacheMeta(meta.cache) ||
    typeof meta.fallback.active !== "boolean" ||
    !isComparisonFallbackCodeOrNull(meta.fallback.code) ||
    typeof meta.fallback.retryable !== "boolean"
  ) {
    return false;
  }

  if (data.status === "fallback") {
    return (
      data.analysis === null &&
      isComparisonFallback(data.fallback) &&
      meta.fallback.active === true &&
      meta.fallback.code === data.fallback.code &&
      meta.fallback.retryable === data.fallback.retryable &&
      meta.output_sha256 === null &&
      meta.cache.status !== "stale"
    );
  }

  return (
    (data.status === "completed" || data.status === "insufficient_data") &&
    data.fallback === null &&
    isCompanyComparisonLLMOutput(data.analysis, requestedTaxIds, data.comparison) &&
    data.analysis.status === data.status &&
    typeof meta.output_sha256 === "string" &&
    meta.fallback.active === false &&
    meta.fallback.code === null &&
    meta.fallback.retryable === false
  );
}

function isCompanyComparisonLLMOutput(
  value: unknown,
  requestedTaxIds: string[],
  comparison: CompanyComparisonResponse,
): value is CompanyComparisonLLMOutput {
  if (
    !isRecord(value) ||
    value.schema_version !== "1.0" ||
    (value.status !== "completed" && value.status !== "insufficient_data") ||
    !isNonEmptyString(value.headline) ||
    !isNonEmptyString(value.overall_observation) ||
    !Array.isArray(value.company_observations) ||
    value.company_observations.length > 9 ||
    !value.company_observations.every((item) =>
      isComparisonCompanyObservation(item, requestedTaxIds),
    ) ||
    !Array.isArray(value.comparison_observations) ||
    value.comparison_observations.length > 4 ||
    !value.comparison_observations.every(isComparisonCrossObservation) ||
    !Array.isArray(value.verification_items) ||
    value.verification_items.length < 1 ||
    value.verification_items.length > 10 ||
    !value.verification_items.every(isComparisonVerificationItem) ||
    !Array.isArray(value.limitations) ||
    value.limitations.length < 3 ||
    value.limitations.length > 10 ||
    !value.limitations.every(isComparisonAnalysisLimitation) ||
    !isRecord(value.provenance) ||
    value.disclaimer !== COMPANY_COMPARISON_AI_DISCLAIMER
  ) {
    return false;
  }
  const provenance = value.provenance;
  if (!(
    provenance.input_schema_version === "1.0" &&
    provenance.prompt_version === "1.0" &&
    Array.isArray(provenance.requested_tax_ids) &&
    sameOrderedStrings(provenance.requested_tax_ids, requestedTaxIds) &&
    provenance.comparison_version === "1.0" &&
    provenance.bizscore_version === "1.0" &&
    provenance.benchmark_catalog_version ===
      comparison.data.context.benchmark_catalog_version &&
    provenance.data_as_of === comparison.data.context.benchmark_as_of &&
    provenance.disclaimer_version === "1.0" &&
    isComparisonEvidencePaths(provenance.evidence_paths_used, 1, 50)
  )) {
    return false;
  }
  const companyTopics = value.company_observations.map(
    (item) => `${item.tax_id}:${item.topic}`,
  );
  const comparisonTopics = value.comparison_observations.map((item) => item.topic);
  const limitationCodes = value.limitations.map((item) => item.code);
  const requiredLimitations = ["public_data_only", "ai_generated", "not_ranked"];
  if (
    (value.status === "completed" &&
      value.company_observations.length === 0 &&
      value.comparison_observations.length === 0) ||
    new Set(companyTopics).size !== companyTopics.length ||
    new Set(comparisonTopics).size !== comparisonTopics.length ||
    new Set(limitationCodes).size !== limitationCodes.length ||
    !requiredLimitations.every((code) => limitationCodes.includes(code))
  ) {
    return false;
  }
  const referencedPaths = [
    ...value.company_observations.flatMap((item) => item.evidence_paths),
    ...value.comparison_observations.flatMap((item) => item.evidence_paths),
    ...value.verification_items.flatMap((item) => item.related_evidence_paths),
    ...value.limitations.flatMap((item) => item.related_evidence_paths),
  ];
  return (
    sameStringSet(referencedPaths, provenance.evidence_paths_used) &&
    provenance.evidence_paths_used.every(
      (path) => resolveJsonPointer({ comparison }, path) !== undefined,
    )
  );
}

function isComparisonCompanyObservation(
  value: unknown,
  requestedTaxIds: string[],
): boolean {
  return (
    isRecord(value) &&
    typeof value.tax_id === "string" &&
    requestedTaxIds.includes(value.tax_id) &&
    isComparisonFindingTopic(value.topic) &&
    isNonEmptyString(value.title) &&
    isNonEmptyString(value.observation) &&
    isComparisonEvidencePaths(value.evidence_paths, 1, 6) &&
    isNonEmptyString(value.caveat)
  );
}

function isComparisonCrossObservation(value: unknown): boolean {
  return (
    isRecord(value) &&
    isComparisonObservationTopic(value.topic) &&
    isNonEmptyString(value.title) &&
    isNonEmptyString(value.observation) &&
    isComparisonEvidencePaths(value.evidence_paths, 1, 8) &&
    isNonEmptyString(value.caveat)
  );
}

function isComparisonVerificationItem(value: unknown): boolean {
  return (
    isRecord(value) &&
    (value.priority === "優先" || value.priority === "一般" || value.priority === "補充") &&
    isNonEmptyString(value.question) &&
    isNonEmptyString(value.reason) &&
    isComparisonEvidencePaths(value.related_evidence_paths, 0, 8)
  );
}

function isComparisonAnalysisLimitation(value: unknown): boolean {
  return (
    isRecord(value) &&
    isComparisonLimitationCode(value.code) &&
    isNonEmptyString(value.message) &&
    isComparisonEvidencePaths(value.related_evidence_paths, 0, 8)
  );
}

function isComparisonEvidencePaths(
  value: unknown,
  minimum = 0,
  maximum = Number.POSITIVE_INFINITY,
): value is string[] {
  return (
    Array.isArray(value) &&
    value.length >= minimum &&
    value.length <= maximum &&
    new Set(value).size === value.length &&
    value.every(
      (path) =>
        typeof path === "string" &&
        /^\/comparison\/(data|meta)(\/[A-Za-z0-9_~-]+)*$/.test(path),
    )
  );
}

function isComparisonCacheMeta(value: Record<string, unknown>): boolean {
  if (!(
    (value.status === "hit" ||
      value.status === "miss" ||
      value.status === "bypass" ||
      value.status === "stale") &&
    value.key_version === "1.0" &&
    isSha256(value.key_sha256) &&
    isIntegerInRange(value.ttl_seconds, 1, Number.MAX_SAFE_INTEGER) &&
    isNonNegativeInteger(value.stale_if_error_seconds) &&
    isStringOrNull(value.cached_at) &&
    isStringOrNull(value.expires_at) &&
    isStringOrNull(value.stale_expires_at)
  )) {
    return false;
  }
  const timestamps = [value.cached_at, value.expires_at, value.stale_expires_at];
  const allEmpty = timestamps.every((item) => item === null);
  const allPresent = timestamps.every(isOffsetDateTime);
  if (!allEmpty && !allPresent) return false;
  if ((value.status === "hit" || value.status === "stale") && !allPresent) {
    return false;
  }
  return (
    !allPresent ||
    (Date.parse(value.cached_at as string) <= Date.parse(value.expires_at as string) &&
      Date.parse(value.expires_at as string) <=
        Date.parse(value.stale_expires_at as string))
  );
}

function isComparisonFallback(value: unknown): value is CompanyComparisonFallback {
  return (
    isRecord(value) &&
    isComparisonFallbackCode(value.code) &&
    isNonEmptyString(value.title) &&
    isNonEmptyString(value.message) &&
    typeof value.retryable === "boolean"
  );
}

function isComparisonFallbackCode(value: unknown): boolean {
  return (
    value === "not_configured" ||
    value === "timeout" ||
    value === "provider_unavailable" ||
    value === "provider_error" ||
    value === "invalid_output"
  );
}

function isComparisonFallbackCodeOrNull(value: unknown): boolean {
  return value === null || isComparisonFallbackCode(value);
}

function isComparisonFindingTopic(value: unknown): boolean {
  return (
    value === "registration_status" ||
    value === "company_age" ||
    value === "registered_capital_scale" ||
    value === "registration_change_recency" ||
    value === "peer_relative_position" ||
    value === "data_completeness"
  );
}

function isComparisonObservationTopic(value: unknown): boolean {
  return (
    value === "bizscore_context" ||
    value === "peer_scope" ||
    value === "data_completeness"
  );
}

function isComparisonLimitationCode(value: unknown): boolean {
  return (
    value === "public_data_only" ||
    value === "ai_generated" ||
    value === "not_ranked" ||
    value === "partial_source_data" ||
    value === "provisional_score" ||
    value === "no_numeric_score" ||
    value === "cross_industry_comparison" ||
    value === "benchmark_unavailable"
  );
}

function isTokenUsageOrNull(value: unknown): boolean {
  return (
    value === null ||
    (isRecord(value) &&
      isNonNegativeInteger(value.input_tokens) &&
      isNonNegativeInteger(value.output_tokens) &&
      isNonNegativeInteger(value.total_tokens))
  );
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

function sameStringSet(value: string[], expected: string[]): boolean {
  return (
    new Set(value).size === new Set(expected).size &&
    value.every((item) => expected.includes(item)) &&
    expected.every((item) => value.includes(item))
  );
}

function isSha256(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isOffsetDateTime(value: unknown): value is string {
  return (
    typeof value === "string" &&
    /(?:Z|[+-]\d{2}:\d{2})$/.test(value) &&
    !Number.isNaN(Date.parse(value))
  );
}

function isNonNegativeInteger(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) >= 0;
}

function isIntegerInRange(
  value: unknown,
  minimum: number,
  maximum: number,
): value is number {
  return (
    Number.isInteger(value) &&
    (value as number) >= minimum &&
    (value as number) <= maximum
  );
}

function sameOrderedStrings(value: unknown[], expected: string[]): boolean {
  return (
    value.length === expected.length &&
    value.every((item, index) => item === expected[index])
  );
}

function isCompanyBizScoreResponse(
  value: unknown,
  expectedTaxId: string,
): value is CompanyBizScoreResponse {
  return (
    isRecord(value) &&
    isRecord(value.data) &&
    isCompanyData(value.data.company, expectedTaxId) &&
    isBizScoreResult(value.data.bizscore) &&
    isRecord(value.data.bizscore) &&
    hasValidBizScoreV1Invariants(
      value.data.bizscore,
      value.data.company.status.code,
      value.data.company.status.description,
    ) &&
    isResponseMeta(value.meta)
  );
}

function isCompanyComparisonResponse(
  value: unknown,
  requestedTaxIds: string[],
): value is CompanyComparisonResponse {
  if (!isRecord(value) || !isRecord(value.data) || !isRecord(value.meta)) {
    return false;
  }
  const { data, meta } = value;
  if (
    data.version !== "1.0" ||
    data.disclaimer !== COMPANY_COMPARISON_DISCLAIMER ||
    !Array.isArray(data.items) ||
    data.items.length < 2 ||
    data.items.length > 3 ||
    !isRecord(data.context) ||
    data.context.ordering !== "request_order" ||
    data.context.tie_handling !== "not_ranked" ||
    typeof data.context.benchmark_catalog_version !== "string" ||
    typeof data.context.benchmark_as_of !== "string" ||
    !isPeerComparisonScope(data.context.peer_comparison_scope) ||
    typeof data.context.same_primary_industry !== "boolean"
  ) {
    return false;
  }
  if (
    meta.comparison_version !== "1.0" ||
    typeof meta.generated_at !== "string" ||
    !Array.isArray(meta.requested_tax_ids) ||
    !meta.requested_tax_ids.every((taxId) => typeof taxId === "string") ||
    meta.requested_count !== requestedTaxIds.length ||
    meta.returned_count !== requestedTaxIds.length ||
    typeof meta.has_partial_source_data !== "boolean" ||
    typeof meta.has_provisional_scores !== "boolean" ||
    typeof meta.has_unscored_companies !== "boolean" ||
    !Array.isArray(meta.warnings) ||
    !meta.warnings.every(isComparisonWarning)
  ) {
    return false;
  }
  if (
    meta.requested_tax_ids.some(
      (taxId, index) => taxId !== requestedTaxIds[index],
    )
  ) {
    return false;
  }
  return data.items.every((item, index) =>
    isComparisonItem(item, index, requestedTaxIds[index]),
  );
}

function isComparisonItem(
  value: unknown,
  expectedIndex: number,
  expectedTaxId: string,
): boolean {
  return (
    isRecord(value) &&
    value.input_index === expectedIndex &&
    isCompanyData(value.company, expectedTaxId) &&
    isBizScoreResult(value.bizscore) &&
    isComparisonMetrics(value.metrics) &&
    isResponseMeta(value.source_meta)
  );
}

function isCompanyData(
  value: unknown,
  expectedTaxId: string,
): value is CompanyData {
  if (
    !isRecord(value) ||
    value.tax_id !== expectedTaxId ||
    typeof value.name !== "string" ||
    !isRecord(value.status) ||
    !isStringOrNull(value.status.code) ||
    !isStringOrNull(value.status.description) ||
    !isRecord(value.capital) ||
    !isNumberOrNull(value.capital.registered) ||
    !isNumberOrNull(value.capital.paid_in) ||
    typeof value.capital.currency !== "string" ||
    !isStringOrNull(value.established_at) ||
    !isNumberOrNull(value.company_age_years) ||
    !isStringOrNull(value.last_changed_at) ||
    !isStringOrNull(value.responsible_name) ||
    !isStringOrNull(value.address) ||
    !isStringOrNull(value.registration_authority) ||
    !Array.isArray(value.business_items)
  ) {
    return false;
  }
  return value.business_items.every(
    (item) =>
      isRecord(item) &&
      typeof item.sequence === "string" &&
      typeof item.code === "string" &&
      isStringOrNull(item.name),
  );
}

function isBizScoreResult(value: unknown): boolean {
  if (
    !isRecord(value) ||
    value.version !== "1.0" ||
    !isNumberOrNull(value.score) ||
    !isStringOrNull(value.band) ||
    !isFiniteNumber(value.coverage) ||
    typeof value.provisional !== "boolean" ||
    typeof value.as_of !== "string" ||
    !isNumberOrNull(value.status_cap) ||
    !Array.isArray(value.dimensions) ||
    !value.dimensions.every(isBizScoreDimension) ||
    !Array.isArray(value.missing_dimensions) ||
    !value.missing_dimensions.every((item) => typeof item === "string") ||
    !isRecord(value.industry) ||
    typeof value.industry.available !== "boolean" ||
    !isRecordOrNull(value.industry.primary_group) ||
    !Array.isArray(value.industry.groups) ||
    !Array.isArray(value.industry.warnings) ||
    !isRecord(value.benchmark) ||
    typeof value.benchmark.catalog_version !== "string" ||
    typeof value.benchmark.catalog_as_of !== "string" ||
    !isStringOrNull(value.benchmark.industry_code) ||
    !isStringOrNull(value.benchmark.snapshot_version) ||
    !isFiniteNumber(value.benchmark.sample_count) ||
    !isRecord(value.peer_benchmark) ||
    !isBizScoreDimension(value.peer_benchmark.dimension) ||
    !isNumberOrNull(value.peer_benchmark.peer_index) ||
    !isFiniteNumber(value.peer_benchmark.sample_size) ||
    !Array.isArray(value.warnings)
  ) {
    return false;
  }
  const primaryGroup = value.industry.primary_group;
  return (
    primaryGroup === null ||
    (typeof primaryGroup.category_code === "string" &&
      typeof primaryGroup.category_name === "string")
  );
}

function hasValidBizScoreV1Invariants(
  value: Record<string, unknown>,
  companyStatusCode: string | null,
  companyStatusDescription: string | null,
): boolean {
  const score = value.score as number | null;
  const coverage = value.coverage as number;
  const statusCap = value.status_cap as number | null;
  const dimensions = value.dimensions as Record<string, unknown>[];
  const missingDimensions = value.missing_dimensions as string[];
  const registrationStatusRule = hasStatusConflict({ code: companyStatusCode, description: companyStatusDescription })
    ? { score: null, statusCap: null }
    : bizScoreRegistrationStatusRuleForCode(companyStatusCode);
  const expectedStatusCap = registrationStatusRule.statusCap;

  if (
    (score !== null && !isIntegerInRange(score, 0, 100)) ||
    !isBizScoreBandOrNull(value.band) ||
    coverage < 0 ||
    coverage > 1 ||
    (statusCap !== null && !isIntegerInRange(statusCap, 0, 100)) ||
    statusCap !== expectedStatusCap ||
    dimensions.length !== BIZSCORE_V1_DIMENSIONS.length
  ) {
    return false;
  }

  let availableWeight = 0;
  let earnedPoints = 0;
  const expectedMissingDimensions: string[] = [];
  for (let index = 0; index < BIZSCORE_V1_DIMENSIONS.length; index += 1) {
    const dimension = dimensions[index];
    const [expectedKey, expectedMaxScore] = BIZSCORE_V1_DIMENSIONS[index];
    const dimensionScore = dimension.score as number | null;
    const available = dimension.available as boolean;

    if (
      dimension.key !== expectedKey ||
      dimension.max_score !== expectedMaxScore ||
      available !== (dimensionScore !== null) ||
      (expectedKey === "registration_status" &&
        dimensionScore !== registrationStatusRule.score) ||
      (dimensionScore !== null &&
        !isIntegerInRange(dimensionScore, 0, expectedMaxScore))
    ) {
      return false;
    }
    if (available) {
      availableWeight += expectedMaxScore;
      earnedPoints += dimensionScore as number;
    } else {
      expectedMissingDimensions.push(expectedKey);
    }
  }

  const expectedScore =
    expectedStatusCap !== null && coverage >= 0.8
      ? Math.min(
          Math.round((earnedPoints / availableWeight) * 100),
          expectedStatusCap,
        )
      : null;
  const expectedBand =
    expectedScore === null ? null : bizScoreBandForScore(expectedScore);

  if (
    !sameOrderedStrings(missingDimensions, expectedMissingDimensions) ||
    Math.abs(coverage - availableWeight / 100) >
      BIZSCORE_COVERAGE_EPSILON ||
    (score === null) !== (value.band === null) ||
    (score === null && value.provisional === true) ||
    (score !== null && coverage < 0.8) ||
    (score !== null && coverage < 1 && value.provisional !== true) ||
    score !== expectedScore ||
    value.band !== expectedBand
  ) {
    return false;
  }

  return true;
}

function bizScoreBandForScore(score: number): string {
  if (score >= 80) return "公開資料呈現較穩健";
  if (score >= 60) return "公開資料呈現一般";
  if (score >= 40) return "建議進一步查核";
  return "需優先查核";
}

function bizScoreRegistrationStatusRuleForCode(
  statusCode: string | null,
): { score: number | null; statusCap: number | null } {
  if (statusCode === "01") return { score: 25, statusCap: 100 };
  if (statusCode !== null && /^(?:0[2-9]|[12]\d|3[0-3])$/.test(statusCode)) {
    return { score: 0, statusCap: null };
  }
  return { score: null, statusCap: null };
}

function isBizScoreDimension(
  value: unknown,
): value is Record<string, unknown> {
  return (
    isRecord(value) &&
    typeof value.key === "string" &&
    typeof value.label === "string" &&
    isNumberOrNull(value.score) &&
    isFiniteNumber(value.max_score) &&
    typeof value.available === "boolean" &&
    Array.isArray(value.evidence) &&
    Array.isArray(value.warnings)
  );
}

function isBizScoreBandOrNull(value: unknown): boolean {
  return (
    value === null ||
    value === "公開資料呈現較穩健" ||
    value === "公開資料呈現一般" ||
    value === "建議進一步查核" ||
    value === "需優先查核"
  );
}

function isComparisonMetrics(value: unknown): boolean {
  return (
    isRecord(value) &&
    isStringOrNull(value.registration_status) &&
    isNumberOrNull(value.company_age_years) &&
    isNumberOrNull(value.registered_capital) &&
    isStringOrNull(value.last_changed_at) &&
    isNumberOrNull(value.bizscore) &&
    isFiniteNumber(value.bizscore_coverage) &&
    typeof value.bizscore_provisional === "boolean" &&
    (value.bizscore_status === "complete" ||
      value.bizscore_status === "provisional" ||
      value.bizscore_status === "unavailable") &&
    isNumberOrNull(value.peer_index) &&
    isStringOrNull(value.industry_code) &&
    Array.isArray(value.missing) &&
    value.missing.every(isComparisonMetricKey)
  );
}

function isResponseMeta(value: unknown): boolean {
  if (!isRecord(value)) return false;
  const freshness = value.data_freshness;
  const fallbackReason = value.fallback_reason;
  const validFreshness =
    freshness === undefined ||
    freshness === "live" ||
    freshness === "fresh_cache" ||
    freshness === "stale_cache";
  const validFallbackReason =
    fallbackReason === undefined || isStringOrNull(fallbackReason);
  const validFreshnessPair =
    freshness === "stale_cache"
      ? typeof fallbackReason === "string"
      : fallbackReason === undefined || fallbackReason === null;
  return (
    typeof value.source === "string" &&
    typeof value.provider === "string" &&
    typeof value.fetched_at === "string" &&
    typeof value.partial === "boolean" &&
    Array.isArray(value.warnings) &&
    value.warnings.every((warning) => typeof warning === "string") &&
    validFreshness &&
    validFallbackReason &&
    validFreshnessPair
  );
}

function isComparisonWarning(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.code === "string" &&
    isStringOrNull(value.tax_id) &&
    typeof value.message === "string"
  );
}

function isComparisonMetricKey(value: unknown): boolean {
  return (
    value === "registration_status" ||
    value === "company_age" ||
    value === "registered_capital" ||
    value === "last_changed_at" ||
    value === "bizscore" ||
    value === "peer_index" ||
    value === "industry"
  );
}

function isPeerComparisonScope(value: unknown): boolean {
  return (
    value === "same_industry_snapshot" ||
    value === "different_industry_snapshots" ||
    value === "unavailable"
  );
}

function isAnalysisFinding(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.topic === "string" &&
    typeof value.title === "string" &&
    typeof value.observation === "string" &&
    Array.isArray(value.evidence_paths) &&
    (value.caveat === null || typeof value.caveat === "string")
  );
}

function isVerificationItem(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.priority === "string" &&
    typeof value.question === "string" &&
    typeof value.reason === "string" &&
    Array.isArray(value.related_evidence_paths)
  );
}

function isAnalysisLimitation(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.code === "string" &&
    typeof value.message === "string" &&
    Array.isArray(value.related_evidence_paths)
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isRecordOrNull(
  value: unknown,
): value is Record<string, unknown> | null {
  return value === null || isRecord(value);
}

function isStringOrNull(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function isNumberOrNull(value: unknown): value is number | null {
  return value === null || isFiniteNumber(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
