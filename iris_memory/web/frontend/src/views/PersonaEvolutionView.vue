<template>
  <div class="persona-evolution-view">
    <ComponentDisabled
      :status="status"
      :error="componentError"
      :error-type="errorType"
      component-name="人格自迭代"
      show-retry
      @retry="refreshState"
    >
      <v-card color="surface" variant="flat" class="iris-hero-card persona-hero mb-3">
        <v-card-text class="pa-4 pa-sm-5">
          <div class="persona-hero__layout">
            <div class="persona-hero__heading">
              <div class="persona-hero__icon" aria-hidden="true">
                <v-icon icon="mdi-account-cog" color="primary" size="28" />
              </div>
              <div>
                <h1 class="persona-hero__title">人格自迭代</h1>
                <p class="persona-hero__subtitle">
                  让 Persona 从真实对话中持续学习。创建任务并选择学习范围，达到条件后系统会安全生成新版本。
                </p>
              </div>
            </div>
            <div class="persona-hero__actions">
              <v-btn
                variant="outlined"
                size="large"
                prepend-icon="mdi-account-switch"
                @click="cloneDialog = true"
              >
                克隆 default
              </v-btn>
              <v-btn
                color="primary"
                variant="flat"
                size="large"
                prepend-icon="mdi-plus"
                class="persona-create-btn"
                @click="openCreate"
              >
                新建迭代任务
              </v-btn>
              <v-tooltip text="刷新数据" location="bottom">
                <template #activator="{ props }">
                  <v-btn
                    v-bind="props"
                    icon="mdi-refresh"
                    variant="text"
                    size="small"
                    aria-label="刷新数据"
                    :loading="loading"
                    @click="loadAll"
                  />
                </template>
              </v-tooltip>
            </div>
          </div>
          <div class="persona-safety-note mt-4">
            <v-icon icon="mdi-shield-check" color="primary" size="18" />
            <span>
              群和用户范围只决定学习语料来源，不改变 Persona 的会话适用范围。默认仅维护
              <code>IRIS_EVOLUTION</code> 受控区块；每次修改都保留完整快照并可非破坏性回滚。
            </span>
          </div>
        </v-card-text>
      </v-card>

      <v-row class="mb-1">
        <v-col cols="6" md="3">
          <div class="iris-stat-box">
            <div class="iris-stat-value">{{ jobs.length }}</div>
            <div class="iris-stat-label">迭代任务</div>
          </div>
        </v-col>
        <v-col cols="6" md="3">
          <div class="iris-stat-box">
            <div class="iris-stat-value">{{ activeJobs }}</div>
            <div class="iris-stat-label">正常运行</div>
          </div>
        </v-col>
        <v-col cols="6" md="3">
          <div class="iris-stat-box">
            <div class="iris-stat-value">{{ pendingCandidates }}</div>
            <div class="iris-stat-label">当前任务待审批</div>
          </div>
        </v-col>
        <v-col cols="6" md="3">
          <div class="iris-stat-box">
            <div class="iris-stat-value">{{ sampleStats?.total ?? 0 }}</div>
            <div class="iris-stat-label">有效语料</div>
          </div>
        </v-col>
      </v-row>

      <v-tabs v-model="tab" color="primary" class="mb-3">
        <v-tab value="jobs">任务与版本</v-tab>
        <v-tab value="samples">语料统计</v-tab>
      </v-tabs>

      <v-window v-model="tab">
        <v-window-item value="jobs">
          <v-card
            v-if="loading && !jobs.length"
            color="surface"
            variant="flat"
            class="iris-card"
          >
            <v-skeleton-loader type="article, actions" />
          </v-card>

          <v-card
            v-else-if="!jobs.length"
            color="surface"
            variant="flat"
            class="iris-card persona-onboarding-card"
          >
            <v-row no-gutters>
              <v-col cols="12" md="7" class="pa-5 pa-sm-6">
                <div class="text-h6 font-weight-bold mb-1">从一个迭代任务开始</div>
                <p class="text-body-2 text-medium-emphasis mb-5">
                  任务会把“学习哪些对话、如何改写 Persona、何时发布”绑定在一起。配置一次，后续可自动运行。
                </p>
                <div class="persona-onboarding-steps">
                  <div class="persona-onboarding-step">
                    <span class="persona-onboarding-step__index">1</span>
                    <div>
                      <div class="font-weight-medium">准备具名 Persona</div>
                      <div class="text-body-2 text-medium-emphasis">已有 Persona 可直接选择；没有时先克隆 default。</div>
                    </div>
                  </div>
                  <div class="persona-onboarding-step">
                    <span class="persona-onboarding-step__index">2</span>
                    <div>
                      <div class="font-weight-medium">设置学习范围与迭代方向</div>
                      <div class="text-body-2 text-medium-emphasis">选择群、用户和审批方式，默认配置适合首次使用。</div>
                    </div>
                  </div>
                  <div class="persona-onboarding-step">
                    <span class="persona-onboarding-step__index">3</span>
                    <div>
                      <div class="font-weight-medium">等待语料积累或手动运行</div>
                      <div class="text-body-2 text-medium-emphasis">每次变更都会生成版本，可审批、查看差异和回滚。</div>
                    </div>
                  </div>
                </div>
              </v-col>

              <v-col cols="12" md="5" class="persona-onboarding-cta pa-5 pa-sm-6">
                <div class="persona-onboarding-cta__icon mb-4" aria-hidden="true">
                  <v-icon icon="mdi-lightning-bolt" size="28" />
                </div>

                <template v-if="personasDegraded">
                  <div class="text-h6 font-weight-bold">暂时无法读取 Persona</div>
                  <div class="text-body-2 text-medium-emphasis mt-2 mb-5">
                    PersonaManager 当前不可用，请刷新后再创建任务。
                  </div>
                  <v-btn color="primary" variant="flat" size="large" block prepend-icon="mdi-refresh" @click="loadAll">
                    刷新重试
                  </v-btn>
                </template>

                <template v-else-if="iterablePersonas.length">
                  <div class="text-h6 font-weight-bold">创建第一个迭代任务</div>
                  <div class="text-body-2 text-medium-emphasis mt-2 mb-5">
                    已检测到 {{ iterablePersonas.length }} 个可用 Persona。推荐先使用“受控区块 + 自动发布”的默认配置。
                  </div>
                  <v-btn color="primary" variant="flat" size="large" block prepend-icon="mdi-plus" @click="openCreate">
                    开始创建
                  </v-btn>
                  <v-btn variant="text" size="small" class="mt-2" block prepend-icon="mdi-account-switch" @click="cloneDialog = true">
                    或克隆 default
                  </v-btn>
                </template>

                <template v-else>
                  <div class="text-h6 font-weight-bold">先准备具名 Persona</div>
                  <div class="text-body-2 text-medium-emphasis mt-2 mb-5">
                    当前没有可迭代的 Persona。先克隆 default，完成后即可创建任务。
                  </div>
                  <v-btn color="primary" variant="flat" size="large" block prepend-icon="mdi-account-switch" @click="cloneDialog = true">
                    克隆 default
                  </v-btn>
                </template>
              </v-col>
            </v-row>
          </v-card>

          <v-row v-else>
            <v-col cols="12" lg="4" xl="3">
              <v-card color="surface" variant="flat" class="iris-card job-list-card">
                <v-card-title class="iris-section-title job-list-title">
                  <v-icon icon="mdi-file-tree" color="primary" />
                  任务
                  <v-spacer />
                  <v-btn
                    color="primary"
                    variant="tonal"
                    size="small"
                    prepend-icon="mdi-plus"
                    @click="openCreate"
                  >
                    新建
                  </v-btn>
                </v-card-title>
                <v-divider />
                <v-list v-if="jobs.length" class="iris-list pa-2" density="compact">
                  <v-list-item
                    v-for="job in jobs"
                    :key="job.id"
                    :active="job.id === selectedJobId"
                    rounded="lg"
                    @click="selectJob(job.id)"
                  >
                    <v-list-item-title class="font-weight-medium">
                      {{ job.name || job.persona_id }}
                    </v-list-item-title>
                    <v-list-item-subtitle>{{ job.persona_id }}</v-list-item-subtitle>
                    <template #append>
                      <v-chip size="x-small" variant="tonal" :color="jobStatusColor(job.status)">
                        {{ jobStatusLabel(job.status) }}
                      </v-chip>
                    </template>
                    <div class="text-caption text-medium-emphasis mt-1">
                      新语料 {{ job.sample_new }} / {{ job.trigger_sample_count }} · 总计 {{ job.sample_total }}
                    </div>
                    <v-progress-linear
                      :model-value="Math.min(100, job.sample_new / Math.max(1, job.trigger_sample_count) * 100)"
                      height="3"
                      rounded
                      color="primary"
                      class="mt-1"
                    />
                  </v-list-item>
                </v-list>
                <div v-else class="iris-empty-state py-8">
                  <v-icon icon="mdi-file-tree" size="48" />
                  <div class="iris-empty-state__title">暂无任务</div>
                  <div class="iris-empty-state__desc">先创建一个具名 Persona 的迭代任务</div>
                </div>
              </v-card>
            </v-col>

            <v-col cols="12" lg="8" xl="9">
              <template v-if="selectedJob">
                <v-card color="surface" variant="flat" class="iris-card mb-3">
                  <v-card-title class="d-flex align-center iris-section-title">
                    <v-icon icon="mdi-account-cog" color="primary" />
                    {{ selectedJob.name || selectedJob.persona_id }}
                    <v-chip size="small" variant="tonal" class="ml-2" :color="jobStatusColor(selectedJob.status)">
                      {{ jobStatusLabel(selectedJob.status) }}
                    </v-chip>
                    <v-spacer />
                    <v-btn size="small" variant="text" prepend-icon="mdi-pencil" @click="openEdit">编辑</v-btn>
                    <v-btn
                      v-if="selectedJob.status === 'active'"
                      size="small"
                      variant="tonal"
                      prepend-icon="mdi-pause"
                      @click="confirmJobAction('pause')"
                    >暂停</v-btn>
                    <v-btn
                      v-else-if="selectedJob.status !== 'conflict'"
                      size="small"
                      variant="tonal"
                      prepend-icon="mdi-play"
                      @click="confirmJobAction('resume')"
                    >恢复</v-btn>
                    <v-btn
                      color="primary"
                      size="small"
                      variant="tonal"
                      prepend-icon="mdi-play"
                      class="ml-2"
                      :loading="actionLoading"
                      :disabled="selectedJob.status !== 'active'"
                      @click="confirmJobAction('run')"
                    >立即迭代</v-btn>
                  </v-card-title>
                  <v-card-text>
                    <v-alert
                      v-if="selectedJob.status === 'conflict'"
                      type="warning"
                      variant="tonal"
                      density="compact"
                      class="mb-3"
                    >
                      检测到 AstrBot 侧外部编辑，自动流程已停止。
                      <v-btn size="small" color="warning" variant="tonal" class="ml-2" @click="confirmJobAction('adopt')">
                        采纳当前 Persona 为新基线
                      </v-btn>
                    </v-alert>
                    <v-row dense>
                      <v-col cols="12" sm="6" md="3">
                        <div class="iris-info-row"><span class="iris-info-row__label">Persona</span><span>{{ selectedJob.persona_id }}</span></div>
                      </v-col>
                      <v-col cols="12" sm="6" md="3">
                        <div class="iris-info-row"><span class="iris-info-row__label">编辑范围</span><span>{{ editModeLabel(selectedJob.edit_mode) }}</span></div>
                      </v-col>
                      <v-col cols="12" sm="6" md="3">
                        <div class="iris-info-row"><span class="iris-info-row__label">审批</span><span>{{ selectedJob.approval_mode === 'auto' ? '自动发布' : '人工审批' }}</span></div>
                      </v-col>
                      <v-col cols="12" sm="6" md="3">
                        <div class="iris-info-row"><span class="iris-info-row__label">上次成功</span><span>{{ formatTime(selectedJob.last_success_at) }}</span></div>
                      </v-col>
                    </v-row>
                    <div class="d-flex flex-wrap ga-2 mt-3">
                      <v-chip size="small" variant="tonal">目标：{{ goalName(selectedJob.goal_preset_id) }}</v-chip>
                      <v-chip size="small" variant="tonal">群：{{ scopeLabel(selectedJob.source_group_ids) }}</v-chip>
                      <v-chip size="small" variant="tonal">用户：{{ scopeLabel(selectedJob.source_user_ids) }}</v-chip>
                      <v-chip size="small" variant="tonal">触发：{{ selectedJob.trigger_sample_count }} 条 + {{ selectedJob.min_interval_hours }} 小时</v-chip>
                      <v-chip v-if="selectedJob.consecutive_failures" size="small" color="error" variant="tonal">
                        连续失败 {{ selectedJob.consecutive_failures }} 次
                      </v-chip>
                    </div>
                  </v-card-text>
                </v-card>

                <v-card color="surface" variant="flat" class="iris-card mb-3">
                  <v-card-title class="iris-section-title">
                    <v-icon icon="mdi-clock-outline" color="primary" />
                    最近运行
                  </v-card-title>
                  <v-table v-if="runs.length" density="compact" class="iris-table">
                    <thead><tr><th>时间</th><th>触发</th><th>状态</th><th>语料</th><th>Token</th><th>结果</th></tr></thead>
                    <tbody>
                      <tr v-for="run in runs" :key="run.id">
                        <td>{{ formatTime(run.started_at) }}</td>
                        <td>{{ run.trigger_type === 'manual' ? '手动' : '自动' }}</td>
                        <td><v-chip size="x-small" variant="tonal" :color="run.status === 'success' ? 'success' : run.status === 'failed' ? 'error' : 'info'">{{ run.status }}</v-chip></td>
                        <td>{{ run.selected_count }} / {{ run.eligible_count }}</td>
                        <td>{{ run.analysis_tokens + run.generation_tokens + run.review_tokens }}</td>
                        <td class="run-result" :title="run.error_message || ''">{{ run.error_code || '—' }}</td>
                      </tr>
                    </tbody>
                  </v-table>
                  <div v-else class="text-body-2 text-medium-emphasis pa-4">暂无运行记录</div>
                </v-card>

                <v-card color="surface" variant="flat" class="iris-card">
                  <v-card-title class="d-flex align-center iris-section-title">
                    <v-icon icon="mdi-backup-restore" color="primary" />
                    Revision 时间线
                    <v-spacer />
                    <span class="text-caption text-medium-emphasis">{{ revisions.length }} 个版本</span>
                  </v-card-title>
                  <v-divider />
                  <v-expansion-panels v-if="revisions.length" variant="accordion" class="revision-panels pa-2">
                    <v-expansion-panel v-for="revision in revisions" :key="revision.id">
                      <v-expansion-panel-title>
                        <div class="d-flex align-center flex-wrap ga-2 w-100 pr-3">
                          <span class="font-weight-medium">v{{ revision.version }}</span>
                          <v-chip size="x-small" variant="tonal" :color="revisionStatusColor(revision.status)">
                            {{ revisionStatusLabel(revision.status) }}
                          </v-chip>
                          <span class="text-caption text-medium-emphasis">{{ formatTime(revision.created_at) }}</span>
                          <span v-if="revision.change_summary?.length" class="revision-summary text-body-2">
                            {{ revision.change_summary.join('；') }}
                          </span>
                        </div>
                      </v-expansion-panel-title>
                      <v-expansion-panel-text>
                        <div class="d-flex flex-wrap ga-2 mb-3">
                          <v-btn
                            v-if="revision.status === 'candidate'"
                            color="success"
                            variant="tonal"
                            size="small"
                            prepend-icon="mdi-check"
                            @click="confirmRevisionAction('approve', revision)"
                          >批准并发布</v-btn>
                          <v-btn
                            v-if="revision.status === 'candidate'"
                            color="error"
                            variant="tonal"
                            size="small"
                            prepend-icon="mdi-close"
                            @click="openReject(revision)"
                          >拒绝</v-btn>
                          <v-btn
                            v-if="canRollback(revision)"
                            color="warning"
                            variant="tonal"
                            size="small"
                            prepend-icon="mdi-backup-restore"
                            @click="confirmRevisionAction('rollback', revision)"
                          >回滚到此版本</v-btn>
                        </div>
                        <v-alert v-if="revision.decision_reason" type="info" variant="tonal" density="compact" class="mb-3">
                          {{ revision.decision_reason }}
                        </v-alert>
                        <div v-if="revision.rationale" class="text-body-2 mb-3">
                          <strong>迭代理由：</strong>{{ revision.rationale }}
                        </div>
                        <GraphemeDiff :before="revision.base_prompt || ''" :after="revision.result_prompt || ''" />
                      </v-expansion-panel-text>
                    </v-expansion-panel>
                  </v-expansion-panels>
                  <div v-else class="iris-empty-state py-8">
                    <v-icon icon="mdi-backup-restore" size="48" />
                    <div class="iris-empty-state__title">尚无人格版本</div>
                  </div>
                </v-card>
              </template>
              <v-card v-else color="surface" variant="flat" class="iris-card">
                <div class="iris-empty-state">
                  <v-icon icon="mdi-hand-pointing-up" size="56" />
                  <div class="iris-empty-state__title">选择一个任务查看详情</div>
                </div>
              </v-card>
            </v-col>
          </v-row>
        </v-window-item>

        <v-window-item value="samples">
          <v-card color="surface" variant="flat" class="iris-card mb-3">
            <v-card-title class="d-flex align-center iris-section-title">
              <v-icon icon="mdi-chart-box" color="primary" />
              语料分布（不展示原文）
              <v-spacer />
              <v-btn color="error" variant="tonal" size="small" prepend-icon="mdi-delete-sweep-outline" @click="confirmClearSamples({})">
                清空全部
              </v-btn>
            </v-card-title>
            <v-card-text>
              <v-alert type="info" variant="tonal" density="compact" class="mb-3">
                此处只显示脱敏语料的计数与分布。删除语料不会删除 Revision 快照，也不会破坏回滚能力。
              </v-alert>
              <v-row>
                <v-col cols="12" md="6">
                  <div class="text-subtitle-1 font-weight-medium mb-2">按群</div>
                  <v-table density="compact" class="iris-table">
                    <thead><tr><th>群</th><th>数量</th><th></th></tr></thead>
                    <tbody>
                      <tr v-for="row in sampleStats?.by_group || []" :key="row.group_id">
                        <td>{{ row.group_name || row.group_id }}</td><td>{{ row.count }}</td>
                        <td class="text-right"><v-btn icon="mdi-delete-outline" size="x-small" variant="text" color="error" @click="confirmClearSamples({ group_id: row.group_id })" /></td>
                      </tr>
                    </tbody>
                  </v-table>
                </v-col>
                <v-col cols="12" md="6">
                  <div class="text-subtitle-1 font-weight-medium mb-2">按用户</div>
                  <v-table density="compact" class="iris-table">
                    <thead><tr><th>用户</th><th>数量</th><th></th></tr></thead>
                    <tbody>
                      <tr v-for="row in sampleStats?.by_user || []" :key="row.user_id">
                        <td>{{ row.user_name || row.user_id }}</td><td>{{ row.count }}</td>
                        <td class="text-right"><v-btn icon="mdi-delete-outline" size="x-small" variant="text" color="error" @click="confirmClearSamples({ user_id: row.user_id })" /></td>
                      </tr>
                    </tbody>
                  </v-table>
                </v-col>
              </v-row>
            </v-card-text>
          </v-card>
        </v-window-item>
      </v-window>
    </ComponentDisabled>

    <v-dialog v-model="jobDialog" max-width="820" persistent>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon icon="mdi-account-cog" color="primary" class="mr-2" />
          {{ editingJobId ? '编辑迭代任务' : '新建迭代任务' }}
        </v-card-title>
        <v-card-text>
          <v-select
            v-if="!editingJobId"
            v-model="form.persona_id"
            :items="iterablePersonas"
            item-title="persona_id"
            item-value="persona_id"
            label="目标 Persona"
            variant="outlined"
            :hint="personasDegraded ? 'PersonaManager 暂不可用，请稍后刷新' : '一个 Persona 只能创建一个任务'"
            persistent-hint
          />
          <v-text-field v-model="form.name" label="任务名称（可选）" variant="outlined" class="mt-3" />
          <v-row>
            <v-col cols="12" md="6">
              <v-select
                v-model="form.goal_preset_id"
                :items="goals"
                item-title="display_name"
                item-value="preset_id"
                label="迭代方向"
                variant="outlined"
              />
            </v-col>
            <v-col cols="12" md="6">
              <v-select v-model="form.approval_mode" :items="approvalModes" label="审批模式" variant="outlined" />
            </v-col>
          </v-row>
          <v-textarea
            v-if="form.goal_preset_id === 'custom'"
            v-model="form.custom_goal"
            label="自定义迭代方向"
            variant="outlined"
            rows="3"
            counter="500"
          />
          <v-select v-model="form.edit_mode" :items="editModes" label="人格编辑范围" variant="outlined" />
          <v-alert v-if="form.edit_mode === 'full_prompt'" type="warning" variant="tonal" density="compact" class="mb-3">
            全人格模式允许重写完整 system prompt，影响所有使用该 Persona 的会话，并会强制独立审查和改动率限制。
            <v-checkbox v-model="fullPromptConfirmed" label="我了解完整人格改写风险" density="compact" hide-details />
          </v-alert>
          <v-combobox
            v-model="form.source_group_ids"
            :items="sampleGroupIds"
            label="学习来源群 ID（留空 = 全部群）"
            multiple chips closable-chips variant="outlined"
          />
          <v-combobox
            v-model="form.source_user_ids"
            :items="sampleUserIds"
            label="学习来源用户 ID（留空 = 匹配群中的全部真人用户）"
            multiple chips closable-chips variant="outlined"
          />
          <v-row>
            <v-col cols="12" md="6"><v-text-field v-model.number="form.trigger_sample_count" type="number" min="1" label="自动触发新增有效消息数" variant="outlined" /></v-col>
            <v-col cols="12" md="6"><v-text-field v-model.number="form.min_interval_hours" type="number" min="1" max="720" label="两次成功迭代最短间隔（小时）" variant="outlined" /></v-col>
          </v-row>
          <v-combobox v-model="form.protected_fragments" label="保护片段（可选）" multiple chips closable-chips variant="outlined" />
          <v-row>
            <v-col cols="12" md="6"><v-text-field v-model="form.provider_id" label="生成 Provider ID（留空使用模块配置）" variant="outlined" /></v-col>
            <v-col cols="12" md="6"><v-text-field v-model="form.reviewer_provider_id" label="审查 Provider ID（留空使用模块配置）" variant="outlined" /></v-col>
          </v-row>
        </v-card-text>
        <v-card-actions>
          <v-spacer /><v-btn variant="text" @click="jobDialog = false">取消</v-btn>
          <v-btn color="primary" :loading="actionLoading" :disabled="!jobFormValid" @click="saveJob">保存</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <v-dialog v-model="cloneDialog" max-width="500">
      <v-card>
        <v-card-title>克隆 default Persona</v-card-title>
        <v-card-text>
          <v-alert type="info" variant="tonal" density="compact" class="mb-3">只创建具名 Persona，不自动改变任何会话绑定。</v-alert>
          <v-text-field v-model="clonePersonaId" label="新 Persona ID" variant="outlined" maxlength="64" />
        </v-card-text>
        <v-card-actions><v-spacer /><v-btn variant="text" @click="cloneDialog = false">取消</v-btn><v-btn color="primary" :disabled="!clonePersonaId.trim()" :loading="actionLoading" @click="clonePersona">克隆</v-btn></v-card-actions>
      </v-card>
    </v-dialog>

    <v-dialog v-model="confirmDialog" max-width="520">
      <v-card>
        <v-card-title class="d-flex align-center"><v-icon icon="mdi-alert" color="warning" class="mr-2" />确认操作</v-card-title>
        <v-card-text>{{ confirmMessage }}</v-card-text>
        <v-card-actions><v-spacer /><v-btn variant="text" @click="confirmDialog = false">取消</v-btn><v-btn color="warning" variant="tonal" :loading="actionLoading" @click="executeConfirmed">确认</v-btn></v-card-actions>
      </v-card>
    </v-dialog>

    <v-dialog v-model="rejectDialog" max-width="520">
      <v-card>
        <v-card-title>拒绝候选 v{{ targetRevision?.version }}</v-card-title>
        <v-card-text><v-textarea v-model="rejectReason" label="拒绝理由（可选）" variant="outlined" counter="500" /></v-card-text>
        <v-card-actions><v-spacer /><v-btn variant="text" @click="rejectDialog = false">取消</v-btn><v-btn color="error" variant="tonal" :loading="actionLoading" @click="rejectRevision">拒绝</v-btn></v-card-actions>
      </v-card>
    </v-dialog>

    <v-snackbar v-model="snackbar" :color="snackbarColor" location="top" :timeout="4000">{{ snackbarText }}</v-snackbar>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import ComponentDisabled from '@/components/ComponentDisabled.vue'
import GraphemeDiff from '@/components/persona-evolution/GraphemeDiff.vue'
import { useComponentState } from '@/composables/useComponentState'
import {
  adoptEvolutionConflict,
  approveEvolutionRevision,
  clearEvolutionSamples,
  cloneDefaultPersona,
  createEvolutionJob,
  getEvolutionGoals,
  getEvolutionJob,
  getEvolutionJobs,
  getEvolutionPersonas,
  getEvolutionRevisions,
  getEvolutionSampleStats,
  pauseEvolutionJob,
  rejectEvolutionRevision,
  resumeEvolutionJob,
  rollbackEvolutionRevision,
  runEvolutionJob,
  updateEvolutionJob
} from '@/api'
import type {
  EvolutionJob,
  EvolutionJobInput,
  EvolutionPersona,
  EvolutionRevision,
  EvolutionRun,
  EvolutionSampleStats
} from '@/types'

const { status, error: componentError, errorType, refreshState } = useComponentState('persona_evolution')
const tab = ref('jobs')
const loading = ref(false)
const actionLoading = ref(false)
const jobs = ref<EvolutionJob[]>([])
const personas = ref<EvolutionPersona[]>([])
const personasDegraded = ref(false)
const goals = ref<Array<{ preset_id: string; display_name: string; text: string }>>([])
const sampleStats = ref<EvolutionSampleStats | null>(null)
const selectedJobId = ref<number | null>(null)
const selectedJob = ref<EvolutionJob | null>(null)
const runs = ref<EvolutionRun[]>([])
const revisions = ref<EvolutionRevision[]>([])
const jobDialog = ref(false)
const editingJobId = ref<number | null>(null)
const fullPromptConfirmed = ref(false)
const cloneDialog = ref(false)
const clonePersonaId = ref('')
const confirmDialog = ref(false)
const confirmMessage = ref('')
const confirmedAction = ref<null | (() => Promise<void>)>(null)
const rejectDialog = ref(false)
const rejectReason = ref('')
const targetRevision = ref<EvolutionRevision | null>(null)
const snackbar = ref(false)
const snackbarText = ref('')
const snackbarColor = ref('success')

const blankForm = (): EvolutionJobInput => ({
  persona_id: '', name: '', goal_preset_id: 'natural', custom_goal: '',
  source_group_ids: [], source_user_ids: [], edit_mode: 'managed_block', approval_mode: 'auto',
  trigger_sample_count: 100, min_interval_hours: 24, provider_id: '', reviewer_provider_id: '', protected_fragments: []
})
const form = reactive<EvolutionJobInput>(blankForm())

const approvalModes = [
  { title: '自动发布（默认）', value: 'auto' },
  { title: '人工审批', value: 'manual' }
]
const editModes = [
  { title: '仅 IRIS_EVOLUTION 受控区块（推荐）', value: 'managed_block' },
  { title: '完整人格', value: 'full_prompt' }
]

const activeJobs = computed(() => jobs.value.filter(job => job.status === 'active').length)
const pendingCandidates = computed(() => revisions.value.filter(rev => rev.status === 'candidate').length)
const iterablePersonas = computed(() => personas.value.filter(p => p.iterable && (!p.has_job || p.persona_id === form.persona_id)))
const sampleGroupIds = computed(() => (sampleStats.value?.by_group || []).map(row => row.group_id).filter(Boolean) as string[])
const sampleUserIds = computed(() => (sampleStats.value?.by_user || []).map(row => row.user_id).filter(Boolean) as string[])
const jobFormValid = computed(() => {
  const personaValid = Boolean(editingJobId.value || form.persona_id?.trim())
  const goalValid = form.goal_preset_id !== 'custom' || Boolean(form.custom_goal.trim())
  const modeValid = form.edit_mode !== 'full_prompt' || fullPromptConfirmed.value
  return personaValid && goalValid && modeValid && form.trigger_sample_count > 0 && form.min_interval_hours > 0
})

function notify(message: string, color = 'success') {
  snackbarText.value = message
  snackbarColor.value = color
  snackbar.value = true
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

async function loadAll() {
  loading.value = true
  try {
    const [jobList, personaResult, goalList, stats] = await Promise.all([
      getEvolutionJobs(), getEvolutionPersonas(), getEvolutionGoals(), getEvolutionSampleStats()
    ])
    jobs.value = jobList
    personas.value = personaResult.personas
    personasDegraded.value = personaResult.degraded
    goals.value = goalList
    sampleStats.value = stats
    if (selectedJobId.value && jobList.some(job => job.id === selectedJobId.value)) {
      await selectJob(selectedJobId.value)
    } else if (jobList.length) {
      await selectJob(jobList[0].id)
    } else {
      selectedJobId.value = null
      selectedJob.value = null
      runs.value = []
      revisions.value = []
    }
  } catch (error) {
    notify(errorMessage(error), 'error')
  } finally {
    loading.value = false
  }
}

async function selectJob(jobId: number) {
  selectedJobId.value = jobId
  try {
    const [detail, timeline] = await Promise.all([getEvolutionJob(jobId), getEvolutionRevisions(jobId)])
    selectedJob.value = detail.job
    runs.value = detail.runs
    revisions.value = timeline
    const index = jobs.value.findIndex(job => job.id === jobId)
    if (index >= 0) jobs.value[index] = detail.job
  } catch (error) {
    notify(errorMessage(error), 'error')
  }
}

function setForm(source: EvolutionJobInput) {
  Object.assign(form, blankForm(), source)
}

function openCreate() {
  editingJobId.value = null
  fullPromptConfirmed.value = false
  setForm(blankForm())
  jobDialog.value = true
}

function openEdit() {
  if (!selectedJob.value) return
  editingJobId.value = selectedJob.value.id
  fullPromptConfirmed.value = selectedJob.value.edit_mode === 'full_prompt'
  setForm({
    name: selectedJob.value.name,
    goal_preset_id: selectedJob.value.goal_preset_id,
    custom_goal: selectedJob.value.custom_goal,
    source_group_ids: [...selectedJob.value.source_group_ids],
    source_user_ids: [...selectedJob.value.source_user_ids],
    edit_mode: selectedJob.value.edit_mode,
    approval_mode: selectedJob.value.approval_mode,
    trigger_sample_count: selectedJob.value.trigger_sample_count,
    min_interval_hours: selectedJob.value.min_interval_hours,
    provider_id: selectedJob.value.provider_id,
    reviewer_provider_id: selectedJob.value.reviewer_provider_id,
    protected_fragments: [...selectedJob.value.protected_fragments]
  })
  jobDialog.value = true
}

function normalizedForm(): EvolutionJobInput {
  const normalize = (items: unknown[]) => [...new Set(items.map(item => String(item).trim()).filter(Boolean))]
  return {
    ...form,
    persona_id: form.persona_id?.trim(),
    name: form.name.trim(),
    custom_goal: form.custom_goal.trim(),
    source_group_ids: normalize(form.source_group_ids),
    source_user_ids: normalize(form.source_user_ids),
    provider_id: form.provider_id.trim(),
    reviewer_provider_id: form.reviewer_provider_id.trim(),
    protected_fragments: normalize(form.protected_fragments)
  }
}

async function saveJob() {
  actionLoading.value = true
  try {
    const payload = normalizedForm()
    if (editingJobId.value) {
      const { persona_id: _personaId, ...editable } = payload
      await updateEvolutionJob(editingJobId.value, editable)
      notify('迭代任务已更新')
    } else {
      await createEvolutionJob(payload)
      notify('迭代任务已创建；历史语料不会立刻触发自动发布')
    }
    jobDialog.value = false
    await loadAll()
  } catch (error) {
    notify(errorMessage(error), 'error')
  } finally {
    actionLoading.value = false
  }
}

async function clonePersona() {
  actionLoading.value = true
  try {
    const message = await cloneDefaultPersona(clonePersonaId.value.trim())
    notify(message)
    cloneDialog.value = false
    clonePersonaId.value = ''
    const result = await getEvolutionPersonas()
    personas.value = result.personas
    personasDegraded.value = result.degraded
  } catch (error) {
    notify(errorMessage(error), 'error')
  } finally {
    actionLoading.value = false
  }
}

function askConfirmation(message: string, action: () => Promise<void>) {
  confirmMessage.value = message
  confirmedAction.value = action
  confirmDialog.value = true
}

async function executeConfirmed() {
  if (!confirmedAction.value) return
  actionLoading.value = true
  try {
    await confirmedAction.value()
    confirmDialog.value = false
    await loadAll()
  } catch (error) {
    notify(errorMessage(error), 'error')
  } finally {
    actionLoading.value = false
    confirmedAction.value = null
  }
}

function confirmJobAction(action: 'pause' | 'resume' | 'run' | 'adopt') {
  if (!selectedJob.value) return
  const id = selectedJob.value.id
  const actions = {
    pause: { message: '暂停后不会自动迭代，但历史与语料仍会保留。', run: () => pauseEvolutionJob(id) },
    resume: { message: '恢复任务并清除错误熔断计数？', run: () => resumeEvolutionJob(id) },
    run: { message: '立即执行会绕过 100 条与 24 小时自动门槛，但仍要求至少 20 条有效语料并经过全部安全校验。', run: () => runEvolutionJob(id) },
    adopt: { message: '将 AstrBot 当前 Persona 作为新的可信基线，解除冲突；不会回放冲突前候选。', run: () => adoptEvolutionConflict(id) }
  }
  const selected = actions[action]
  askConfirmation(selected.message, async () => {
    const result = await selected.run()
    notify(String(result.message || '操作成功'))
  })
}

function confirmRevisionAction(action: 'approve' | 'rollback', revision: EvolutionRevision) {
  targetRevision.value = revision
  const message = action === 'approve'
    ? `批准 v${revision.version} 后将立即写入 Persona；服务端会重新校验基线哈希和全部发布闸门。`
    : `回滚到 v${revision.version} 会创建一个新的 Revision，不会删除后续历史。`
  askConfirmation(message, async () => {
    if (action === 'approve') await approveEvolutionRevision(revision.id)
    else await rollbackEvolutionRevision(revision.id)
    notify(action === 'approve' ? '候选已批准并发布' : '已创建并应用回滚版本')
  })
}

function openReject(revision: EvolutionRevision) {
  targetRevision.value = revision
  rejectReason.value = ''
  rejectDialog.value = true
}

async function rejectRevision() {
  if (!targetRevision.value) return
  actionLoading.value = true
  try {
    await rejectEvolutionRevision(targetRevision.value.id, rejectReason.value.trim())
    rejectDialog.value = false
    notify('候选已拒绝')
    await loadAll()
  } catch (error) {
    notify(errorMessage(error), 'error')
  } finally {
    actionLoading.value = false
  }
}

function confirmClearSamples(scope: { group_id?: string; user_id?: string }) {
  const label = scope.group_id ? `群 ${scope.group_id}` : scope.user_id ? `用户 ${scope.user_id}` : '全部'
  askConfirmation(`确认清除${label}的学习语料？此操作不影响 Revision 历史，但语料原文不可恢复。`, async () => {
    const count = await clearEvolutionSamples(scope)
    notify(`已清除 ${count} 条语料`)
  })
}

const jobStatusLabel = (value: string) => ({ active: '运行中', paused: '已暂停', conflict: '冲突', paused_error: '错误暂停' }[value] || value)
const jobStatusColor = (value: string) => ({ active: 'success', paused: 'default', conflict: 'warning', paused_error: 'error' }[value] || 'default')
const revisionStatusLabel = (value: string) => ({ candidate: '待审批', publishing: '发布中', applied: '已应用', rejected: '已拒绝', failed_validation: '校验失败', publish_failed: '发布失败', external_change: '外部修改', rollback: '回滚', no_change: '无变化' }[value] || value)
const revisionStatusColor = (value: string) => ({ candidate: 'warning', publishing: 'info', applied: 'success', rejected: 'default', failed_validation: 'error', publish_failed: 'error', external_change: 'warning', rollback: 'info', no_change: 'default' }[value] || 'default')
const editModeLabel = (value: string) => value === 'managed_block' ? '受控区块' : '完整人格'
const goalName = (id: string) => goals.value.find(goal => goal.preset_id === id)?.display_name || id
const scopeLabel = (items: string[]) => items.length ? items.join(', ') : '不限'
const canRollback = (revision: EvolutionRevision) => ['applied', 'rollback'].includes(revision.status)
const formatTime = (value: number | null | undefined) => value ? new Date(value * 1000).toLocaleString('zh-CN', { hour12: false }) : '—'

const handleGlobalRefresh = () => loadAll()
onMounted(() => {
  loadAll()
  window.addEventListener('iris:refresh', handleGlobalRefresh)
})
onUnmounted(() => window.removeEventListener('iris:refresh', handleGlobalRefresh))
</script>

<style scoped>
.persona-hero {
  overflow: hidden;
}
.persona-hero__layout {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
}
.persona-hero__heading {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  min-width: 0;
}
.persona-hero__icon {
  display: grid;
  flex: 0 0 auto;
  width: 48px;
  height: 48px;
  place-items: center;
  border-radius: 14px;
  background: rgba(var(--v-theme-primary), 0.12);
}
.persona-hero__title {
  margin: 0 0 5px;
  color: rgba(var(--v-theme-on-surface), 0.92);
  font-size: 1.35rem;
  font-weight: 700;
  line-height: 1.3;
}
.persona-hero__subtitle {
  max-width: 620px;
  margin: 0;
  color: rgba(var(--v-theme-on-surface), 0.62);
  font-size: 0.9rem;
  line-height: 1.6;
}
.persona-hero__actions {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 10px;
}
.persona-create-btn {
  box-shadow: 0 6px 16px rgba(var(--v-theme-primary), 0.24) !important;
}
.persona-safety-note {
  display: flex;
  align-items: flex-start;
  gap: 9px;
  padding: 10px 12px;
  color: rgba(var(--v-theme-on-surface), 0.68);
  background: rgba(var(--v-theme-primary), 0.055);
  border: 1px solid rgba(var(--v-theme-primary), 0.1);
  border-radius: 9px;
  font-size: 0.8rem;
  line-height: 1.55;
}
.persona-safety-note .v-icon {
  flex: 0 0 auto;
  margin-top: 1px;
}
.persona-onboarding-card {
  overflow: hidden;
}
.persona-onboarding-steps {
  display: grid;
  gap: 18px;
}
.persona-onboarding-step {
  display: grid;
  grid-template-columns: 32px minmax(0, 1fr);
  align-items: start;
  gap: 12px;
}
.persona-onboarding-step__index {
  display: grid;
  width: 30px;
  height: 30px;
  place-items: center;
  color: rgb(var(--v-theme-primary));
  background: rgba(var(--v-theme-primary), 0.12);
  border-radius: 50%;
  font-size: 0.78rem;
  font-weight: 700;
}
.persona-onboarding-cta {
  display: flex;
  flex-direction: column;
  justify-content: center;
  background: linear-gradient(
    145deg,
    rgba(var(--v-theme-primary), 0.12),
    rgba(var(--v-theme-primary), 0.035)
  );
  border-left: 1px solid rgba(var(--v-theme-primary), 0.1);
}
.persona-onboarding-cta__icon {
  display: grid;
  width: 52px;
  height: 52px;
  place-items: center;
  color: rgb(var(--v-theme-on-primary));
  background: rgb(var(--v-theme-primary));
  border-radius: 15px;
  box-shadow: 0 6px 16px rgba(var(--v-theme-primary), 0.22);
}
.job-list-title :deep(.v-btn .v-icon) {
  margin-right: 4px;
}
.job-list-card { min-height: 320px; }
.job-list-card :deep(.v-list-item) { margin-bottom: 6px; border: 1px solid rgba(var(--v-theme-on-surface), 0.06); }
.job-list-card :deep(.v-list-item--active) { border-color: rgba(var(--v-theme-primary), 0.3); }
.revision-summary { flex: 1 1 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.revision-panels :deep(.v-expansion-panel) { border: 1px solid rgba(var(--v-theme-on-surface), 0.08); margin-bottom: 6px; }
.run-result { max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
code { color: rgb(var(--v-theme-primary)); }
@media (max-width: 1080px) {
  .persona-hero__layout {
    align-items: stretch;
    flex-direction: column;
  }
  .persona-hero__actions {
    flex-wrap: wrap;
  }
}
@media (max-width: 700px) {
  .persona-evolution-view :deep(.v-card-title) { flex-wrap: wrap; gap: 6px; }
  .persona-hero__actions > :deep(.v-btn:not(.v-btn--icon)) {
    flex: 1 1 210px;
  }
  .persona-onboarding-cta {
    border-top: 1px solid rgba(var(--v-theme-primary), 0.1);
    border-left: 0;
  }
}
</style>
