import { apiGet, apiPost } from './request'
import type {
  EvolutionGoalPreset,
  EvolutionJob,
  EvolutionJobInput,
  EvolutionPersona,
  EvolutionRevision,
  EvolutionRun,
  EvolutionSampleStats
} from '@/types'

interface BaseResponse {
  success: boolean
  error?: string
  error_code?: string
  message?: string
}

export class EvolutionApiError extends Error {
  errorCode?: string

  constructor(message: string, errorCode?: string) {
    super(message)
    this.name = 'EvolutionApiError'
    this.errorCode = errorCode
  }
}

function assertSuccess(response: BaseResponse, fallback: string): void {
  if (!response.success) {
    throw new EvolutionApiError(response.error || response.message || fallback, response.error_code)
  }
}

export async function getEvolutionGoals(): Promise<EvolutionGoalPreset[]> {
  const response = await apiGet<any>('persona-evolution/goals')
  assertSuccess(response, '获取迭代目标失败')
  return response.goals || []
}

export async function getEvolutionPersonas(): Promise<{ personas: EvolutionPersona[]; degraded: boolean }> {
  const response = await apiGet<any>('persona-evolution/personas')
  assertSuccess(response, '获取 Persona 列表失败')
  return { personas: response.personas || [], degraded: Boolean(response.degraded) }
}

export async function cloneDefaultPersona(personaId: string): Promise<string> {
  const response = await apiPost<any>('persona-evolution/personas/clone-default', { persona_id: personaId })
  assertSuccess(response, '克隆 default Persona 失败')
  return response.message || `已克隆为 ${personaId}`
}

export async function getEvolutionJobs(): Promise<EvolutionJob[]> {
  const response = await apiGet<any>('persona-evolution/jobs')
  assertSuccess(response, '获取迭代任务失败')
  return response.jobs || []
}

export async function getEvolutionJob(jobId: number): Promise<{
  job: EvolutionJob
  runs: EvolutionRun[]
  revisions: EvolutionRevision[]
}> {
  const response = await apiGet<any>(`persona-evolution/jobs/${jobId}`)
  assertSuccess(response, '获取任务详情失败')
  return { job: response.job, runs: response.runs || [], revisions: response.revisions || [] }
}

export async function createEvolutionJob(input: EvolutionJobInput): Promise<EvolutionJob> {
  const response = await apiPost<any>('persona-evolution/jobs', input)
  assertSuccess(response, '创建迭代任务失败')
  return response.job
}

export async function updateEvolutionJob(jobId: number, input: Omit<EvolutionJobInput, 'persona_id'>): Promise<EvolutionJob> {
  const response = await apiPost<any>(`persona-evolution/jobs/${jobId}/update`, input)
  assertSuccess(response, '更新迭代任务失败')
  return response.job
}

async function jobAction(jobId: number, action: string, fallback: string): Promise<BaseResponse & Record<string, unknown>> {
  const response = await apiPost<any>(`persona-evolution/jobs/${jobId}/${action}`)
  assertSuccess(response, fallback)
  return response
}

export const pauseEvolutionJob = (jobId: number) => jobAction(jobId, 'pause', '暂停任务失败')
export const resumeEvolutionJob = (jobId: number) => jobAction(jobId, 'resume', '恢复任务失败')
export const runEvolutionJob = (jobId: number) => jobAction(jobId, 'run', '执行迭代失败')
export const adoptEvolutionConflict = (jobId: number) =>
  jobAction(jobId, 'conflict/adopt-current', '采纳当前 Persona 失败')

export async function getEvolutionRevisions(jobId: number): Promise<EvolutionRevision[]> {
  const response = await apiGet<any>(`persona-evolution/jobs/${jobId}/revisions`, { limit: 100 })
  assertSuccess(response, '获取版本时间线失败')
  return response.revisions || []
}

async function revisionAction(revisionId: number, action: string, body?: Record<string, unknown>): Promise<BaseResponse> {
  const response = await apiPost<any>(`persona-evolution/revisions/${revisionId}/${action}`, body)
  assertSuccess(response, `${action} Revision 失败`)
  return response
}

export const approveEvolutionRevision = (revisionId: number) => revisionAction(revisionId, 'approve')
export const rejectEvolutionRevision = (revisionId: number, reason: string) =>
  revisionAction(revisionId, 'reject', { reason })
export const rollbackEvolutionRevision = (revisionId: number) => revisionAction(revisionId, 'rollback')

export async function getEvolutionSampleStats(): Promise<EvolutionSampleStats> {
  const response = await apiGet<any>('persona-evolution/samples/stats')
  assertSuccess(response, '获取语料统计失败')
  return response.stats
}

export async function clearEvolutionSamples(scope: { group_id?: string; user_id?: string }): Promise<number> {
  const response = await apiPost<any>('persona-evolution/samples/clear', scope)
  assertSuccess(response, '清除语料失败')
  return response.deleted || 0
}
