export interface CompanyStatus {
  code: string | null;
  description: string | null;
}

export interface CompanyCapital {
  registered: number | null;
  paid_in: number | null;
  currency: string;
}

export interface BusinessItem {
  sequence: string;
  code: string;
  name: string | null;
}

export interface CompanyData {
  tax_id: string;
  name: string;
  status: CompanyStatus;
  capital: CompanyCapital;
  established_at: string | null;
  company_age_years: number | null;
  last_changed_at: string | null;
  responsible_name: string | null;
  address: string | null;
  registration_authority: string | null;
  business_items: BusinessItem[];
}

export interface ResponseMeta {
  source: string;
  provider: string;
  fetched_at: string;
  partial: boolean;
  warnings: string[];
  data_freshness: "live" | "fresh_cache" | "stale_cache";
  fallback_reason: string | null;
}

export interface CompanyResponse {
  data: CompanyData;
  meta: ResponseMeta;
}

export interface BizScoreDimension {
  key: string;
  label: string;
  score: number | null;
  max_score: number;
  available: boolean;
  evidence: string[];
  warnings: string[];
}

export interface IndustryGroup {
  category_code: string;
  category_name: string;
  source_sequence: string;
  source_business_item_code: string;
  source_business_item_name: string | null;
  benchmark_eligible: boolean;
}

export interface IndustryClassification {
  version: string;
  available: boolean;
  primary_group: IndustryGroup | null;
  groups: IndustryGroup[];
  excluded_business_item_codes: string[];
  unmapped_business_item_codes: string[];
  warnings: string[];
}

export interface BenchmarkReference {
  catalog_version: string;
  catalog_as_of: string;
  catalog_industry_mapping_version: string;
  classification_version: string;
  industry_code: string | null;
  snapshot_version: string | null;
  snapshot_source_version: string | null;
  snapshot_checksum_sha256: string | null;
  sample_count: number;
}

export interface PeerBenchmark {
  dimension: BizScoreDimension;
  industry_code: string | null;
  benchmark_version: string | null;
  sample_size: number;
  excluded_sample_count: number;
  minimum_sample_size: number;
  age_percentile: number | null;
  capital_percentile: number | null;
  peer_index: number | null;
  company_age_median: number | null;
  registered_capital_median: number | null;
}

export type BizScoreBand =
  | "公開資料呈現較穩健"
  | "公開資料呈現一般"
  | "建議進一步查核"
  | "需優先查核";

export interface BizScoreResult {
  version: "1.0";
  score: number | null;
  band: BizScoreBand | null;
  coverage: number;
  provisional: boolean;
  as_of: string;
  status_cap: number | null;
  dimensions: BizScoreDimension[];
  missing_dimensions: string[];
  industry: IndustryClassification;
  benchmark: BenchmarkReference;
  peer_benchmark: PeerBenchmark;
  warnings: string[];
}

export interface CompanyBizScoreResponse {
  data: {
    company: CompanyData;
    bizscore: BizScoreResult;
  };
  meta: ResponseMeta;
}

export const COMPANY_COMPARISON_DISCLAIMER =
  "公司比較僅並列政府公開登記資料與 BizScore v1 指標；順序依使用者選取，不代表信用、付款或履約能力、投資價值或合作建議。" as const;

export type ComparisonMetricKey =
  | "registration_status"
  | "company_age"
  | "registered_capital"
  | "last_changed_at"
  | "bizscore"
  | "peer_index"
  | "industry";

export type PeerComparisonScope =
  | "same_industry_snapshot"
  | "different_industry_snapshots"
  | "unavailable";

export interface CompanyComparisonMetrics {
  registration_status: string | null;
  company_age_years: number | null;
  registered_capital: number | null;
  last_changed_at: string | null;
  bizscore: number | null;
  bizscore_coverage: number;
  bizscore_provisional: boolean;
  bizscore_status: "complete" | "provisional" | "unavailable";
  peer_index: number | null;
  industry_code: string | null;
  missing: ComparisonMetricKey[];
}

export interface CompanyComparisonItem {
  input_index: number;
  company: CompanyData;
  bizscore: BizScoreResult;
  metrics: CompanyComparisonMetrics;
  source_meta: ResponseMeta;
}

export interface CompanyComparisonWarning {
  code: string;
  tax_id: string | null;
  message: string;
}

export interface CompanyComparisonResponse {
  data: {
    version: "1.0";
    items: CompanyComparisonItem[];
    context: {
      ordering: "request_order";
      tie_handling: "not_ranked";
      benchmark_catalog_version: string;
      benchmark_as_of: string;
      peer_comparison_scope: PeerComparisonScope;
      same_primary_industry: boolean;
    };
    disclaimer: typeof COMPANY_COMPARISON_DISCLAIMER;
  };
  meta: {
    comparison_version: "1.0";
    generated_at: string;
    requested_tax_ids: string[];
    requested_count: number;
    returned_count: number;
    has_partial_source_data: boolean;
    has_provisional_scores: boolean;
    has_unscored_companies: boolean;
    warnings: CompanyComparisonWarning[];
  };
}

export const COMPANY_COMPARISON_AI_DISCLAIMER =
  "AI 比較說明由模型依政府公開登記資料與 BizScore v1 產生，僅供合作前查核問題整理；順序不是排名，不代表信用、付款或履約能力，也不是合作、投資、授信或法律建議。" as const;

export type CompanyComparisonAnalysisStatus =
  | "completed"
  | "insufficient_data"
  | "fallback";

export type CompanyComparisonAnalysisFindingTopic =
  | "registration_status"
  | "company_age"
  | "registered_capital_scale"
  | "registration_change_recency"
  | "peer_relative_position"
  | "data_completeness";

export type CompanyComparisonObservationTopic =
  | "bizscore_context"
  | "peer_scope"
  | "data_completeness";

export type CompanyComparisonEvidencePath = `/comparison/${string}`;

export interface CompanyComparisonCompanyObservation {
  tax_id: string;
  topic: CompanyComparisonAnalysisFindingTopic;
  title: string;
  observation: string;
  evidence_paths: CompanyComparisonEvidencePath[];
  caveat: string;
}

export interface CompanyComparisonCrossObservation {
  topic: CompanyComparisonObservationTopic;
  title: string;
  observation: string;
  evidence_paths: CompanyComparisonEvidencePath[];
  caveat: string;
}

export interface CompanyComparisonVerificationItem {
  priority: VerificationPriority;
  question: string;
  reason: string;
  related_evidence_paths: CompanyComparisonEvidencePath[];
}

export type CompanyComparisonLimitationCode =
  | "public_data_only"
  | "ai_generated"
  | "not_ranked"
  | "partial_source_data"
  | "provisional_score"
  | "no_numeric_score"
  | "cross_industry_comparison"
  | "benchmark_unavailable";

export interface CompanyComparisonAnalysisLimitation {
  code: CompanyComparisonLimitationCode;
  message: string;
  related_evidence_paths: CompanyComparisonEvidencePath[];
}

export interface CompanyComparisonAnalysisProvenance {
  input_schema_version: "1.0";
  prompt_version: "1.0";
  requested_tax_ids: string[];
  comparison_version: "1.0";
  bizscore_version: "1.0";
  benchmark_catalog_version: string;
  data_as_of: string;
  disclaimer_version: "1.0";
  evidence_paths_used: CompanyComparisonEvidencePath[];
}

export interface CompanyComparisonLLMOutput {
  schema_version: "1.0";
  status: "completed" | "insufficient_data";
  headline: string;
  overall_observation: string;
  company_observations: CompanyComparisonCompanyObservation[];
  comparison_observations: CompanyComparisonCrossObservation[];
  verification_items: CompanyComparisonVerificationItem[];
  limitations: CompanyComparisonAnalysisLimitation[];
  provenance: CompanyComparisonAnalysisProvenance;
  disclaimer: typeof COMPANY_COMPARISON_AI_DISCLAIMER;
}

export type CompanyComparisonFallbackCode =
  | "not_configured"
  | "timeout"
  | "provider_unavailable"
  | "provider_error"
  | "invalid_output";

export interface CompanyComparisonFallback {
  code: CompanyComparisonFallbackCode;
  title: string;
  message: string;
  retryable: boolean;
}

export type CompanyComparisonCacheStatus =
  | "hit"
  | "miss"
  | "bypass"
  | "stale";

export interface CompanyComparisonAnalysisCacheMeta {
  status: CompanyComparisonCacheStatus;
  key_version: "1.0";
  key_sha256: string;
  ttl_seconds: number;
  stale_if_error_seconds: number;
  cached_at: string | null;
  expires_at: string | null;
  stale_expires_at: string | null;
}

export interface CompanyComparisonAnalysisMeta {
  analysis_version: "1.0";
  provider: string;
  model: string;
  provider_response_id: string | null;
  provider_request_id: string | null;
  service_tier: string | null;
  generated_at: string;
  prompt_version: "1.0";
  system_prompt_sha256: string;
  input_schema_version: "1.0";
  output_schema_version: "1.0";
  input_sha256: string;
  output_sha256: string | null;
  attempts: number;
  duration_ms: number;
  usage: LLMTokenUsage | null;
  cache: CompanyComparisonAnalysisCacheMeta;
  fallback: {
    active: boolean;
    code: CompanyComparisonFallbackCode | null;
    retryable: boolean;
  };
}

export interface CompanyComparisonAnalysisResponse {
  data: {
    status: CompanyComparisonAnalysisStatus;
    comparison: CompanyComparisonResponse;
    analysis: CompanyComparisonLLMOutput | null;
    fallback: CompanyComparisonFallback | null;
  };
  meta: CompanyComparisonAnalysisMeta;
}

export interface CompanySearchItem {
  tax_id: string;
  name: string;
  status: CompanyStatus;
  registered_capital: number | null;
  established_at: string | null;
  last_changed_at: string | null;
}

export interface CompanySearchResponse {
  data: CompanySearchItem[];
  meta: ResponseMeta;
}

export interface ApiErrorDetail {
  code: string;
  message: string;
  retryable: boolean;
}

export interface CompanyAnalysisLLMInput {
  schema_version: "1.0";
  task: "company_analysis";
  language: "zh-TW";
  disclaimer_version: "1.0";
  company: CompanyData;
  bizscore: BizScoreResult;
  source_meta: ResponseMeta;
}

export type EvidencePath = `/${"company" | "bizscore" | "source_meta"}/${string}`;

export type AnalysisFindingTopic =
  | "registration_status"
  | "company_age"
  | "registered_capital_scale"
  | "registration_change_recency"
  | "peer_relative_position"
  | "data_completeness";

export interface LLMAnalysisFinding {
  topic: AnalysisFindingTopic;
  title: string;
  observation: string;
  evidence_paths: EvidencePath[];
  caveat: string | null;
}

export type VerificationPriority = "優先" | "一般" | "補充";

export interface LLMVerificationItem {
  priority: VerificationPriority;
  question: string;
  reason: string;
  related_evidence_paths: EvidencePath[];
}

export type AnalysisLimitationCode =
  | "public_data_only"
  | "ai_generated"
  | "partial_source_data"
  | "missing_dimension"
  | "provisional_score"
  | "no_numeric_score"
  | "benchmark_unavailable";

export interface LLMAnalysisLimitation {
  code: AnalysisLimitationCode;
  message: string;
  related_evidence_paths: EvidencePath[];
}

export interface LLMAnalysisProvenance {
  input_schema_version: "1.0";
  prompt_version: "1.0";
  company_tax_id: string;
  bizscore_version: "1.0";
  benchmark_catalog_version: string;
  data_as_of: string;
  disclaimer_version: "1.0";
  evidence_paths_used: EvidencePath[];
}

export interface CompanyAnalysisLLMOutput {
  schema_version: "1.0";
  status: "completed" | "insufficient_data";
  headline: string;
  overall_observation: string;
  findings: LLMAnalysisFinding[];
  verification_items: LLMVerificationItem[];
  limitations: LLMAnalysisLimitation[];
  provenance: LLMAnalysisProvenance;
  disclaimer: "AI 內容僅解釋已提供資料與計算結果，可能有錯誤或遺漏，不應取代獨立查核與專業意見。";
}

export interface LLMTokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface CompanyAnalysisMeta {
  provider: string;
  model: string;
  provider_response_id: string | null;
  provider_request_id: string | null;
  service_tier: string | null;
  generated_at: string;
  prompt_version: "1.0";
  system_prompt_sha256: string;
  input_schema_version: "1.0";
  output_schema_version: "1.0";
  input_sha256: string;
  output_sha256: string;
  attempts: number;
  duration_ms: number;
  usage: LLMTokenUsage | null;
}

export interface CompanyAnalysisResponse {
  data: {
    input: CompanyAnalysisLLMInput;
    analysis: CompanyAnalysisLLMOutput;
  };
  meta: CompanyAnalysisMeta;
}
