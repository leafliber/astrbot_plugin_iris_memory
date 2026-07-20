<template>
  <div class="profile-view">
    <ComponentDisabled
      :status="status"
      :error="error"
      :error-type="errorType"
      component-name="画像管理"
      @retry="refreshState"
    >
      <div class="d-flex align-center flex-wrap ga-2 mb-3">
        <div class="d-flex align-center">
          <v-icon icon="mdi-account-cog" color="primary" class="mr-2" />
          <span class="text-h6">画像管理</span>
        </div>
        <v-spacer />
        <IsolationBadge
          type="profile"
          :enabled="isolationStatus.enable_group_isolation"
        />
        <IsolationBadge
          type="persona"
          :enabled="isolationStatus.enable_persona_isolation"
        />
      </div>

      <v-tabs v-model="activeTab" color="primary" align-tabs="start">
        <v-tab value="group">
          <v-icon icon="mdi-account-group" class="mr-1" />
          群聊画像
        </v-tab>
        <v-tab value="user">
          <v-icon icon="mdi-account" class="mr-1" />
          用户画像
        </v-tab>
      </v-tabs>

      <v-window v-model="activeTab" class="mt-4">
        <v-window-item value="group">
          <v-row>
            <v-col cols="12" md="4">
              <v-card color="surface" variant="flat" class="iris-list-card">
                <v-card-title class="d-flex align-center iris-section-title">
                  <v-icon icon="mdi-account-group" color="primary" class="mr-2" />
                  群聊列表
                  <v-spacer />
                  <v-btn
                    icon="mdi-refresh"
                    variant="text"
                    size="small"
                    :loading="profileStore.groupListLoading"
                    @click="loadGroupList"
                  />
                </v-card-title>
                <v-card-text class="pa-0">
                  <v-text-field
                    v-model="groupSearchQuery"
                    placeholder="搜索群聊..."
                    prepend-inner-icon="mdi-magnify"
                    variant="outlined"
                    density="compact"
                    hide-details
                    class="ma-2"
                    clearable
                  />
                  <v-progress-linear
                    v-if="profileStore.groupListLoading"
                    indeterminate
                    color="primary"
                  />

                  <v-list v-else-if="filteredGroupList.length > 0" lines="two" class="iris-list py-0">
                    <v-list-item
                      v-for="group in filteredGroupList"
                      :key="group.group_id"
                      :active="selectedGroupId === group.group_id"
                      @click="selectGroup(group.group_id)"
                    >
                      <template #prepend>
                        <v-avatar color="primary" variant="tonal" size="36">
                          <v-icon icon="mdi-account-group" size="20" />
                        </v-avatar>
                      </template>

                      <v-list-item-title>{{ group.group_name || group.group_id }}</v-list-item-title>
                      <v-list-item-subtitle>
                        <v-icon icon="mdi-tag" size="small" class="mr-1" />
                        {{ group.group_id }}
                      </v-list-item-subtitle>
                    </v-list-item>
                  </v-list>

                  <div v-else class="iris-empty-state">
                    <v-icon icon="mdi-account-group-outline" size="48" />
                    <div class="iris-empty-state__title">{{ groupSearchQuery ? '未找到匹配的群聊' : '暂无群聊数据' }}</div>
                  </div>
                </v-card-text>
              </v-card>
            </v-col>

            <v-col cols="12" md="8">
              <v-card color="surface" variant="flat" class="iris-card">
                <v-card-title class="d-flex align-center iris-section-title">
                  <v-icon icon="mdi-information" color="primary" class="mr-2" />
                  群聊画像详情
                  <v-spacer />
                  <v-btn
                    v-if="selectedGroupId"
                    icon="mdi-delete-outline"
                    variant="text"
                    size="small"
                    color="error"
                    @click="confirmDeleteGroup"
                  />
                  <v-btn
                    v-if="selectedGroupId"
                    icon="mdi-refresh"
                    variant="text"
                    size="small"
                    :loading="profileStore.groupProfileLoading"
                    @click="loadGroupProfile"
                  />
                </v-card-title>
                <v-card-text>
                  <template v-if="selectedGroupId">
                    <v-progress-linear
                      v-if="profileStore.groupProfileLoading"
                      indeterminate
                      color="primary"
                      class="mb-4"
                    />

                    <div v-else-if="profileStore.currentGroupProfile" class="profile-content">
                      <div class="profile-header mb-4">
                        <v-avatar color="primary" size="56" class="mr-4">
                          <v-icon icon="mdi-account-group" size="32" />
                        </v-avatar>
                        <div>
                          <div class="d-flex align-center">
                            <div class="text-h5 mr-2">{{ profileStore.currentGroupProfile.group_name || '未命名群聊' }}</div>
                            <v-btn icon="mdi-pencil" variant="text" size="x-small" @click="startEditGroupField('group_name')" />
                          </div>
                          <div class="text-caption text-medium-emphasis">{{ profileStore.currentGroupProfile.group_id }}</div>
                        </div>
                      </div>

                      <v-card variant="outlined" class="iris-card iris-card-hover info-card">
                        <v-card-title class="text-subtitle-2 pb-0 d-flex align-center">
                          <v-icon icon="mdi-emoticon-outline" color="accent" class="mr-2" />
                          群聊氛围
                          <v-spacer />
                          <v-btn icon="mdi-plus" variant="text" size="x-small" @click="startAddTag('group', 'atmosphere_tags')" />
                        </v-card-title>
                        <v-card-text>
                          <div v-if="profileStore.currentGroupProfile.atmosphere_tags?.length" class="tags-container">
                            <v-chip
                              v-for="tag in profileStore.currentGroupProfile.atmosphere_tags"
                              :key="tag"
                              color="accent"
                              variant="tonal"
                              size="small"
                              class="ma-1"
                              closable
                              @click:close="removeTagFromGroup('atmosphere_tags', tag)"
                            >
                              {{ tag }}
                            </v-chip>
                          </div>
                          <div v-else class="text-medium-emphasis text-body-2">暂无氛围标签</div>
                        </v-card-text>
                      </v-card>

                      <v-card variant="outlined" class="iris-card iris-card-hover info-card mt-4">
                        <v-card-title class="text-subtitle-2 pb-0 d-flex align-center">
                          <v-icon icon="mdi-heart" color="pink" class="mr-2" />
                          兴趣偏好
                          <v-spacer />
                          <v-btn icon="mdi-plus" variant="text" size="x-small" @click="startAddTag('group', 'interests')" />
                        </v-card-title>
                        <v-card-text>
                          <div v-if="profileStore.currentGroupProfile.interests?.length" class="tags-container">
                            <v-chip
                              v-for="interest in profileStore.currentGroupProfile.interests"
                              :key="interest"
                              color="pink"
                              variant="tonal"
                              size="small"
                              class="ma-1"
                              closable
                              @click:close="removeTagFromGroup('interests', interest)"
                            >
                              {{ interest }}
                            </v-chip>
                          </div>
                          <div v-else class="text-medium-emphasis text-body-2">暂无兴趣标签</div>
                        </v-card-text>
                      </v-card>

                      <v-card variant="outlined" class="iris-card iris-card-hover info-card mt-4">
                        <v-card-title class="text-subtitle-2 pb-0 d-flex align-center">
                          <v-icon icon="mdi-star" color="warning" class="mr-2" />
                          核心特征
                          <v-spacer />
                          <v-btn icon="mdi-plus" variant="text" size="x-small" @click="startAddTag('group', 'long_term_tags')" />
                        </v-card-title>
                        <v-card-text>
                          <div v-if="profileStore.currentGroupProfile.long_term_tags?.length" class="tags-container">
                            <v-chip
                              v-for="tag in profileStore.currentGroupProfile.long_term_tags"
                              :key="tag"
                              color="warning"
                              variant="tonal"
                              size="small"
                              class="ma-1"
                              closable
                              @click:close="removeTagFromGroup('long_term_tags', tag)"
                            >
                              {{ tag }}
                            </v-chip>
                          </div>
                          <div v-else class="text-medium-emphasis text-body-2">暂无数据</div>
                        </v-card-text>
                      </v-card>

                      <v-card variant="outlined" class="iris-card iris-card-hover info-card mt-4">
                        <v-card-title class="text-subtitle-2 pb-0 d-flex align-center">
                          <v-icon icon="mdi-block-helper" color="error" class="mr-2" />
                          禁忌话题
                          <v-spacer />
                          <v-btn icon="mdi-plus" variant="text" size="x-small" @click="startAddTag('group', 'blacklist_topics')" />
                        </v-card-title>
                        <v-card-text>
                          <div v-if="profileStore.currentGroupProfile.blacklist_topics?.length" class="tags-container">
                            <v-chip
                              v-for="topic in profileStore.currentGroupProfile.blacklist_topics"
                              :key="topic"
                              color="error"
                              variant="tonal"
                              size="small"
                              class="ma-1"
                              closable
                              @click:close="removeTagFromGroup('blacklist_topics', topic)"
                            >
                              {{ topic }}
                            </v-chip>
                          </div>
                          <div v-else class="text-medium-emphasis text-body-2">暂无禁忌话题</div>
                        </v-card-text>
                      </v-card>
                    </div>

                    <div v-else class="iris-empty-state">
                      <v-icon icon="mdi-file-document-outline" size="56" />
                      <div class="iris-empty-state__title">暂无群聊画像数据</div>
                    </div>
                  </template>

                  <div v-else class="iris-empty-state">
                    <v-icon icon="mdi-hand-pointing-up" size="56" />
                    <div class="iris-empty-state__title">请从左侧选择一个群聊</div>
                  </div>
                </v-card-text>
              </v-card>
            </v-col>
          </v-row>
        </v-window-item>

        <v-window-item value="user">
          <v-row>
            <v-col cols="12">
              <v-alert
                :type="isolationStatus.enable_group_isolation ? 'success' : 'info'"
                variant="tonal"
                density="compact"
                class="mb-2"
              >
                <div class="text-body-2">
                  <v-icon icon="mdi-information-outline" class="mr-1" />
                  <template v-if="isolationStatus.enable_group_isolation">
                    群聊隔离已开启：用户画像按真实群聊ID存储，可使用下方筛选器按群聊查看。
                  </template>
                  <template v-else>
                    群聊隔离已关闭：所有用户画像统一存储于「全局」命名空间（group_id=default），不同群聊的用户画像共享。
                  </template>
                </div>
              </v-alert>
            </v-col>
          </v-row>

          <v-row>
            <v-col cols="12" md="4">
              <v-card color="surface" variant="flat" class="iris-list-card">
                <v-card-title class="d-flex align-center iris-section-title">
                  <v-icon icon="mdi-account" color="secondary" class="mr-2" />
                  用户列表
                  <v-spacer />
                  <v-btn
                    icon="mdi-refresh"
                    variant="text"
                    size="small"
                    :loading="profileStore.userListLoading"
                    @click="loadUserList"
                  />
                </v-card-title>
                <v-card-text class="pa-0">
                  <v-select
                    v-model="userGroupIdFilter"
                    :items="userGroupOptions"
                    item-title="title"
                    item-value="value"
                    placeholder="全部群聊"
                    prepend-inner-icon="mdi-account-group"
                    variant="outlined"
                    density="compact"
                    hide-details
                    class="ma-2"
                    clearable
                  />
                  <v-text-field
                    v-model="userSearchQuery"
                    placeholder="搜索用户..."
                    prepend-inner-icon="mdi-magnify"
                    variant="outlined"
                    density="compact"
                    hide-details
                    class="mx-2 mb-1"
                    clearable
                  />
                  <v-progress-linear
                    v-if="profileStore.userListLoading"
                    indeterminate
                    color="primary"
                  />

                  <v-list v-else-if="filteredUserList.length > 0" lines="two" class="iris-list py-0">
                    <v-list-item
                      v-for="user in filteredUserList"
                      :key="(user.group_id || 'default') + ':' + user.user_id"
                      :active="selectedUserId === user.user_id && selectedUserGroupId === (user.group_id || 'default')"
                      @click="selectUser(user.user_id, user.group_id || 'default')"
                    >
                      <template #prepend>
                        <v-avatar color="secondary" variant="tonal" size="36">
                          <v-icon icon="mdi-account" size="20" />
                        </v-avatar>
                      </template>

                      <v-list-item-title>{{ user.nickname || user.user_id }}</v-list-item-title>
                      <v-list-item-subtitle>
                        <v-icon icon="mdi-account-group" size="small" class="mr-1" />
                        {{ displayGroupId(user.group_id) }}
                      </v-list-item-subtitle>
                    </v-list-item>
                  </v-list>

                  <div v-else class="iris-empty-state">
                    <v-icon icon="mdi-account-outline" size="48" />
                    <div class="iris-empty-state__title">{{ userSearchQuery ? '未找到匹配的用户' : '暂无用户数据' }}</div>
                  </div>
                </v-card-text>
              </v-card>
            </v-col>

            <v-col cols="12" md="8">
              <v-card color="surface" variant="flat" class="iris-card">
                <v-card-title class="d-flex align-center iris-section-title">
                  <v-icon icon="mdi-account-details" color="secondary" class="mr-2" />
                  用户画像详情
                  <v-spacer />
                  <v-btn
                    v-if="selectedUserId"
                    icon="mdi-delete-outline"
                    variant="text"
                    size="small"
                    color="error"
                    @click="confirmDeleteUser"
                  />
                  <v-btn
                    v-if="selectedUserId"
                    icon="mdi-refresh"
                    variant="text"
                    size="small"
                    :loading="profileStore.userProfileLoading"
                    @click="loadUserProfile"
                  />
                </v-card-title>
                <v-card-text>
                  <template v-if="selectedUserId">
                    <v-progress-linear
                      v-if="profileStore.userProfileLoading"
                      indeterminate
                      color="primary"
                      class="mb-4"
                    />

                    <div v-else-if="profileStore.currentUserProfile" class="profile-content">
                      <div class="profile-header mb-4">
                        <v-avatar color="secondary" size="56" class="mr-4">
                          <v-icon icon="mdi-account" size="32" />
                        </v-avatar>
                        <div class="flex-grow-1">
                          <div class="d-flex align-center">
                            <div class="text-h5 mr-2">{{ profileStore.currentUserProfile.user_name || '未命名用户' }}</div>
                            <v-btn icon="mdi-pencil" variant="text" size="x-small" @click="startEditUserField('user_name')" />
                          </div>
                          <div class="d-flex align-center flex-wrap ga-2 mt-1">
                            <div class="text-caption text-medium-emphasis">{{ profileStore.currentUserProfile.user_id }}</div>
                            <v-chip
                              size="x-small"
                              variant="tonal"
                              color="secondary"
                              label
                            >
                              <v-icon icon="mdi-account-group" size="small" class="mr-1" />
                              {{ displayGroupId(selectedUserGroupId) }}
                            </v-chip>
                          </div>
                        </div>
                      </div>

                      <v-card variant="outlined" class="iris-card iris-hero-card mb-4 favorability-card">
                        <v-card-text class="pa-4">
                          <div class="d-flex align-center">
                            <v-icon :color="favorabilityColor" size="large" class="mr-3">mdi-heart-pulse</v-icon>
                            <div class="flex-grow-1">
                              <div class="d-flex align-center">
                                <span class="text-subtitle-1 font-weight-medium mr-2">好感度</span>
                                <v-chip :color="favorabilityColor" variant="tonal" size="small" label>
                                  {{ favorabilityLevel }}
                                </v-chip>
                                <v-spacer />
                                <span class="text-h6 font-weight-bold" :class="`text-${favorabilityColor}`">
                                  {{ Math.round(profileStore.currentUserProfile.favorability ?? 0) }}
                                </span>
                                <span class="text-caption text-medium-emphasis ml-1">/ 100</span>
                                <v-btn icon="mdi-pencil" variant="text" size="x-small" class="ml-2"
                                       @click="startEditUserField('favorability')" />
                              </div>
                              <v-progress-linear
                                :model-value="profileStore.currentUserProfile.favorability ?? 0"
                                :color="favorabilityColor"
                                height="10"
                                rounded
                                class="mt-2"
                              />
                            </div>
                          </div>
                        </v-card-text>
                      </v-card>

                      <v-row>
                        <v-col cols="12" sm="6">
                          <v-card variant="outlined" class="iris-card iris-card-hover info-card">
                            <v-card-text>
                              <div class="d-flex align-center mb-2">
                                <v-icon icon="mdi-briefcase" color="primary" size="small" class="mr-2" />
                                <span class="text-caption text-medium-emphasis">职业/身份</span>
                                <v-spacer />
                                <v-btn icon="mdi-pencil" variant="text" size="x-small" @click="startEditUserField('occupation')" />
                              </div>
                              <div class="text-body-1">{{ profileStore.currentUserProfile.occupation || '暂无' }}</div>
                            </v-card-text>
                          </v-card>
                        </v-col>

                        <v-col cols="12" sm="6">
                          <v-card variant="outlined" class="iris-card iris-card-hover info-card">
                            <v-card-text>
                              <div class="d-flex align-center mb-2">
                                <v-icon icon="mdi-translate" color="info" size="small" class="mr-2" />
                                <span class="text-caption text-medium-emphasis">语言风格</span>
                                <v-spacer />
                                <v-btn icon="mdi-pencil" variant="text" size="x-small" @click="startEditUserField('language_style')" />
                              </div>
                              <div class="text-body-1">{{ profileStore.currentUserProfile.language_style || '暂无' }}</div>
                            </v-card-text>
                          </v-card>
                        </v-col>
                      </v-row>

                      <v-row>
                        <v-col cols="12" sm="6">
                          <v-card variant="outlined" class="iris-card iris-card-hover info-card">
                            <v-card-text>
                              <div class="d-flex align-center mb-2">
                                <v-icon icon="mdi-robot" color="accent" size="small" class="mr-2" />
                                <span class="text-caption text-medium-emphasis">与Bot关系</span>
                                <v-spacer />
                                <v-btn icon="mdi-pencil" variant="text" size="x-small" @click="startEditUserField('bot_relationship')" />
                              </div>
                              <div class="text-body-1">{{ profileStore.currentUserProfile.bot_relationship || '暂无' }}</div>
                            </v-card-text>
                          </v-card>
                        </v-col>

                        <v-col cols="12" sm="6">
                          <v-card variant="outlined" class="iris-card iris-card-hover info-card">
                            <v-card-text>
                              <div class="d-flex align-center mb-2">
                                <v-icon icon="mdi-message-text-outline" color="teal" size="small" class="mr-2" />
                                <span class="text-caption text-medium-emphasis">沟通偏好</span>
                                <v-spacer />
                                <v-btn icon="mdi-pencil" variant="text" size="x-small" @click="startEditUserField('communication_style')" />
                              </div>
                              <div class="text-body-1">{{ profileStore.currentUserProfile.communication_style || '暂无' }}</div>
                            </v-card-text>
                          </v-card>
                        </v-col>
                      </v-row>

                      <v-row>
                        <v-col cols="12" sm="6">
                          <v-card variant="outlined" class="iris-card iris-card-hover info-card">
                            <v-card-text>
                              <div class="d-flex align-center mb-2">
                                <v-icon icon="mdi-emoticon-outline" color="orange" size="small" class="mr-2" />
                                <span class="text-caption text-medium-emphasis">情感基线</span>
                                <v-spacer />
                                <v-btn icon="mdi-pencil" variant="text" size="x-small" @click="startEditUserField('emotional_baseline')" />
                              </div>
                              <div class="text-body-1">{{ profileStore.currentUserProfile.emotional_baseline || '暂无' }}</div>
                            </v-card-text>
                          </v-card>
                        </v-col>
                      </v-row>

                      <v-card variant="outlined" class="iris-card iris-card-hover info-card mt-4">
                        <v-card-title class="text-subtitle-2 pb-0 d-flex align-center">
                          <v-icon icon="mdi-account-switch" color="cyan" class="mr-2" />
                          历史曾用名
                          <v-spacer />
                          <v-btn icon="mdi-plus" variant="text" size="x-small" @click="startAddTag('user', 'historical_names')" />
                        </v-card-title>
                        <v-card-text>
                          <div v-if="profileStore.currentUserProfile.historical_names?.length">
                            <v-chip
                              v-for="name in profileStore.currentUserProfile.historical_names"
                              :key="name"
                              color="cyan"
                              variant="tonal"
                              size="small"
                              class="ma-1"
                              closable
                              @click:close="removeTagFromUser('historical_names', name)"
                            >
                              {{ name }}
                            </v-chip>
                          </div>
                          <div v-else class="text-medium-emphasis text-body-2">暂无历史名称</div>
                        </v-card-text>
                      </v-card>

                      <v-card variant="outlined" class="iris-card iris-card-hover info-card mt-4">
                        <v-card-title class="text-subtitle-2 pb-0 d-flex align-center">
                          <v-icon icon="mdi-emoticon-outline" color="purple" class="mr-2" />
                          性格特征
                          <v-spacer />
                          <v-btn icon="mdi-plus" variant="text" size="x-small" @click="startAddTag('user', 'personality_tags')" />
                        </v-card-title>
                        <v-card-text>
                          <div v-if="profileStore.currentUserProfile.personality_tags?.length" class="tags-container">
                            <v-chip
                              v-for="tag in profileStore.currentUserProfile.personality_tags"
                              :key="tag"
                              color="purple"
                              variant="tonal"
                              size="small"
                              class="ma-1"
                              closable
                              @click:close="removeTagFromUser('personality_tags', tag)"
                            >
                              {{ tag }}
                            </v-chip>
                          </div>
                          <div v-else class="text-medium-emphasis text-body-2">暂无性格标签</div>
                        </v-card-text>
                      </v-card>

                      <v-card variant="outlined" class="iris-card iris-card-hover info-card mt-4">
                        <v-card-title class="text-subtitle-2 pb-0 d-flex align-center">
                          <v-icon icon="mdi-heart" color="pink" class="mr-2" />
                          兴趣爱好
                          <v-spacer />
                          <v-btn icon="mdi-plus" variant="text" size="x-small" @click="startAddTag('user', 'interests')" />
                        </v-card-title>
                        <v-card-text>
                          <div v-if="profileStore.currentUserProfile.interests?.length" class="tags-container">
                            <v-chip
                              v-for="interest in profileStore.currentUserProfile.interests"
                              :key="interest"
                              color="pink"
                              variant="tonal"
                              size="small"
                              class="ma-1"
                              closable
                              @click:close="removeTagFromUser('interests', interest)"
                            >
                              {{ interest }}
                            </v-chip>
                          </div>
                          <div v-else class="text-medium-emphasis text-body-2">暂无兴趣标签</div>
                        </v-card-text>
                      </v-card>

                      <v-card variant="outlined" class="iris-card iris-card-hover info-card mt-4">
                        <v-card-title class="text-subtitle-2 pb-0 d-flex align-center">
                          <v-icon icon="mdi-calendar-star" color="warning" class="mr-2" />
                          重要事件
                          <v-spacer />
                          <v-btn icon="mdi-plus" variant="text" size="x-small" @click="startAddTag('user', 'important_events')" />
                        </v-card-title>
                        <v-card-text>
                          <v-list v-if="profileStore.currentUserProfile.important_events?.length" density="compact" class="bg-transparent pa-0">
                            <v-list-item
                              v-for="(event, idx) in profileStore.currentUserProfile.important_events"
                              :key="idx"
                              class="px-0"
                            >
                              <template #prepend>
                                <v-icon icon="mdi-star" color="warning" size="small" />
                              </template>
                              <v-list-item-title>{{ event }}</v-list-item-title>
                              <template #append>
                                <v-btn icon="mdi-close" variant="text" size="x-small" @click="removeTagFromUser('important_events', event)" />
                              </template>
                            </v-list-item>
                          </v-list>
                          <div v-else class="text-medium-emphasis text-body-2">暂无重要事件</div>
                        </v-card-text>
                      </v-card>

                      <v-card variant="outlined" class="iris-card iris-card-hover info-card mt-4">
                        <v-card-title class="text-subtitle-2 pb-0 d-flex align-center">
                          <v-icon icon="mdi-calendar-clock" color="success" class="mr-2" />
                          重要日期
                          <v-spacer />
                          <v-btn icon="mdi-plus" variant="text" size="x-small" @click="showAddDateDialog = true" />
                        </v-card-title>
                        <v-card-text>
                          <v-list v-if="profileStore.currentUserProfile.important_dates?.length" density="compact" class="bg-transparent pa-0">
                            <v-list-item
                              v-for="(item, idx) in profileStore.currentUserProfile.important_dates"
                              :key="idx"
                              class="px-0"
                            >
                              <template #prepend>
                                <v-icon icon="mdi-calendar" color="success" size="small" />
                              </template>
                              <v-list-item-title>{{ item.description }}</v-list-item-title>
                              <v-list-item-subtitle>{{ item.date }}</v-list-item-subtitle>
                              <template #append>
                                <v-btn icon="mdi-close" variant="text" size="x-small" @click="removeDateFromUser(idx)" />
                              </template>
                            </v-list-item>
                          </v-list>
                          <div v-else class="text-medium-emphasis text-body-2">暂无重要日期</div>
                        </v-card-text>
                      </v-card>

                      <v-card variant="outlined" class="iris-card iris-card-hover info-card mt-4">
                        <v-card-title class="text-subtitle-2 pb-0 d-flex align-center">
                          <v-icon icon="mdi-block-helper" color="error" class="mr-2" />
                          禁忌话题
                          <v-spacer />
                          <v-btn icon="mdi-plus" variant="text" size="x-small" @click="startAddTag('user', 'taboo_topics')" />
                        </v-card-title>
                        <v-card-text>
                          <div v-if="profileStore.currentUserProfile.taboo_topics?.length" class="tags-container">
                            <v-chip
                              v-for="topic in profileStore.currentUserProfile.taboo_topics"
                              :key="topic"
                              color="error"
                              variant="tonal"
                              size="small"
                              class="ma-1"
                              closable
                              @click:close="removeTagFromUser('taboo_topics', topic)"
                            >
                              {{ topic }}
                            </v-chip>
                          </div>
                          <div v-else class="text-medium-emphasis text-body-2">暂无禁忌话题</div>
                        </v-card-text>
                      </v-card>

                      <v-card v-if="profileStore.currentUserProfile.custom_fields && Object.keys(profileStore.currentUserProfile.custom_fields).length > 0" variant="outlined" class="info-card mt-4">
                        <v-card-title class="text-subtitle-2 pb-0 d-flex align-center">
                          <v-icon icon="mdi-tag-multiple" color="secondary" class="mr-2" />
                          自定义字段
                        </v-card-title>
                        <v-card-text>
                          <div v-for="(value, key) in profileStore.currentUserProfile.custom_fields" :key="key" class="d-flex align-center mb-2">
                            <span class="text-body-2 font-weight-medium mr-2">{{ key }}</span>
                            <span class="text-body-2 text-medium-emphasis">{{ value }}</span>
                          </div>
                        </v-card-text>
                      </v-card>
                    </div>

                    <div v-else class="iris-empty-state">
                      <v-icon icon="mdi-file-document-outline" size="56" />
                      <div class="iris-empty-state__title">暂无用户画像数据</div>
                    </div>
                  </template>

                  <div v-else class="iris-empty-state">
                    <v-icon icon="mdi-hand-pointing-up" size="56" />
                    <div class="iris-empty-state__title">请从左侧选择一个用户</div>
                  </div>
                </v-card-text>
              </v-card>
            </v-col>
          </v-row>
        </v-window-item>
      </v-window>
    </ComponentDisabled>

    <v-dialog v-model="showEditDialog" max-width="400" class="iris-dialog">
      <v-card>
        <v-card-title>编辑 {{ editFieldLabel }}</v-card-title>
        <v-card-text>
          <v-text-field v-model="editFieldValue" variant="outlined" density="compact" />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showEditDialog = false">取消</v-btn>
          <v-btn color="primary" variant="text" @click="submitEditField">保存</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <v-dialog v-model="showAddTagDialog" max-width="400" class="iris-dialog">
      <v-card>
        <v-card-title>添加{{ addTagFieldLabel }}</v-card-title>
        <v-card-text>
          <v-text-field v-model="addTagValue" variant="outlined" density="compact" placeholder="输入内容后按回车" @keyup.enter="submitAddTag" />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showAddTagDialog = false">取消</v-btn>
          <v-btn color="primary" variant="text" @click="submitAddTag">添加</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <v-dialog v-model="showAddDateDialog" max-width="400" class="iris-dialog">
      <v-card>
        <v-card-title>添加重要日期</v-card-title>
        <v-card-text>
          <v-text-field v-model="addDateValue" variant="outlined" density="compact" label="日期" placeholder="如 01-15" class="mb-2" />
          <v-text-field v-model="addDateDesc" variant="outlined" density="compact" label="描述" placeholder="如 生日" />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showAddDateDialog = false">取消</v-btn>
          <v-btn color="primary" variant="text" @click="submitAddDate">添加</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <v-dialog v-model="showDeleteDialog" max-width="400" class="iris-dialog">
      <v-card>
        <v-card-title>确认删除</v-card-title>
        <v-card-text>
          {{ deleteDialogMessage }}
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showDeleteDialog = false">取消</v-btn>
          <v-btn color="error" variant="text" @click="executeDelete">删除</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <v-snackbar v-model="showSnackbar" :color="snackbarColor" :timeout="2000">
      {{ snackbarText }}
    </v-snackbar>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useProfileStore } from '@/stores'
import { useComponentState } from '@/composables/useComponentState'
import ComponentDisabled from '@/components/ComponentDisabled.vue'
import IsolationBadge from '@/components/IsolationBadge.vue'
import { useIsolationStatus } from '@/composables/useIsolationStatus'
import { getGroupList } from '@/api/profile'
import type { GroupProfile, UserProfile } from '@/types'

const profileStore = useProfileStore()
const { status, error, errorType, refreshState } = useComponentState('profile')
const { status: isolationStatus } = useIsolationStatus()

const activeTab = ref('group')
const groupSearchQuery = ref('')
const userSearchQuery = ref('')
const selectedGroupId = ref<string | null>(null)
const selectedUserId = ref<string | null>(null)
const selectedUserGroupId = ref<string | undefined>(undefined)

// 用户列表群聊筛选（null = 全部群聊）
const userGroupIdFilter = ref<string | null>(null)
// 群聊下拉选项数据
const groups = ref<{ group_id: string; group_name?: string }[]>([])
const userGroupOptions = computed(() => {
  const opts: Array<{ title: string; value: string }> = []
  const isolationOff = !isolationStatus.value.enable_group_isolation
  // 隔离关闭时，用户画像统一存于 "default"；提供 "全局" 选项便于筛选
  if (isolationOff) {
    opts.push({ title: '全局（default）', value: 'default' })
  }
  for (const g of groups.value) {
    // 隔离关闭时跳过重复的 default（group_index 可能含真实群ID，但用户画像不在那里）
    if (isolationOff && g.group_id === 'default') continue
    opts.push({
      title: g.group_name || g.group_id,
      value: g.group_id,
    })
  }
  return opts
})

const fetchGroups = async () => {
  try {
    const list = await getGroupList()
    groups.value = (list || []).map((g: any) => ({
      group_id: g.group_id,
      group_name: g.group_name,
    }))
  } catch (e) {
    console.error('获取群聊列表失败:', e)
    groups.value = []
  }
}

// 展示 group_id：隔离关闭时 "default" 显示为 "全局"
const displayGroupId = (groupId?: string): string => {
  if (!groupId) return '全局'
  if (groupId === 'default' && !isolationStatus.value.enable_group_isolation) return '全局'
  return groupId
}

const showEditDialog = ref(false)
const editFieldType = ref<'group' | 'user'>('group')
const editFieldName = ref('')
const editFieldLabel = ref('')
const editFieldValue = ref('')

const showAddTagDialog = ref(false)
const addTagType = ref<'group' | 'user'>('group')
const addTagField = ref('')
const addTagFieldLabel = ref('')
const addTagValue = ref('')

const showAddDateDialog = ref(false)
const addDateValue = ref('')
const addDateDesc = ref('')

const showDeleteDialog = ref(false)
const deleteDialogType = ref<'group' | 'user'>('group')
const deleteDialogMessage = ref('')

const showSnackbar = ref(false)
const snackbarText = ref('')
const snackbarColor = ref('success')

const FIELD_LABELS: Record<string, string> = {
  group_name: '群聊名称',
  atmosphere_tags: '氛围标签',
  interests: '兴趣偏好',
  long_term_tags: '核心特征',
  blacklist_topics: '禁忌话题',
  user_name: '用户昵称',
  personality_tags: '性格标签',
  occupation: '职业/身份',
  language_style: '语言风格',
  communication_style: '沟通偏好',
  emotional_baseline: '情感基线',
  bot_relationship: '与Bot关系',
  historical_names: '历史曾用名',
  important_events: '重要事件',
  taboo_topics: '禁忌话题',
  favorability: '好感度',
}

const filteredGroupList = computed(() => {
  if (!groupSearchQuery.value) return profileStore.groupList
  const query = groupSearchQuery.value.toLowerCase()
  return profileStore.groupList.filter(g =>
    (g.group_name?.toLowerCase().includes(query)) ||
    g.group_id.toLowerCase().includes(query)
  )
})

const filteredUserList = computed(() => {
  if (!userSearchQuery.value) return profileStore.userList
  const query = userSearchQuery.value.toLowerCase()
  return profileStore.userList.filter(u =>
    (u.nickname?.toLowerCase().includes(query)) ||
    u.user_id.toLowerCase().includes(query) ||
    (u.group_id?.toLowerCase().includes(query))
  )
})

const favorabilityLevel = computed(() => {
  const v = profileStore.currentUserProfile?.favorability ?? 0
  if (v < 20) return '陌生'
  if (v < 40) return '认识'
  if (v < 60) return '熟悉'
  if (v < 80) return '友好'
  return '亲密'
})

const favorabilityColor = computed(() => {
  const v = profileStore.currentUserProfile?.favorability ?? 0
  if (v < 20) return 'grey'
  if (v < 40) return 'blue-grey'
  if (v < 60) return 'info'
  if (v < 80) return 'success'
  return 'pink'
})

const notify = (text: string, color = 'success') => {
  snackbarText.value = text
  snackbarColor.value = color
  showSnackbar.value = true
}

const loadGroupList = () => {
  profileStore.fetchGroupList()
}

const loadUserList = () => {
  profileStore.fetchUserList(userGroupIdFilter.value || undefined)
}

const selectGroup = (groupId: string) => {
  selectedGroupId.value = groupId
  profileStore.fetchGroupProfile(groupId)
}

const selectUser = (userId: string, groupId?: string) => {
  selectedUserId.value = userId
  selectedUserGroupId.value = groupId
  profileStore.fetchUserProfile(userId, groupId)
}

const loadGroupProfile = () => {
  if (selectedGroupId.value) {
    profileStore.fetchGroupProfile(selectedGroupId.value)
  }
}

const loadUserProfile = () => {
  if (selectedUserId.value) {
    profileStore.fetchUserProfile(selectedUserId.value, selectedUserGroupId.value)
  }
}

const startEditGroupField = (field: string) => {
  editFieldType.value = 'group'
  editFieldName.value = field
  editFieldLabel.value = FIELD_LABELS[field] || field
  const profile = profileStore.currentGroupProfile as GroupProfile | null
  editFieldValue.value = String((profile as unknown as Record<string, unknown>)?.[field] ?? '')
  showEditDialog.value = true
}

const startEditUserField = (field: string) => {
  editFieldType.value = 'user'
  editFieldName.value = field
  editFieldLabel.value = FIELD_LABELS[field] || field
  const profile = profileStore.currentUserProfile as UserProfile | null
  editFieldValue.value = String((profile as unknown as Record<string, unknown>)?.[field] ?? '')
  showEditDialog.value = true
}

const submitEditField = async () => {
  try {
    let value: string | number = editFieldValue.value
    if (editFieldName.value === 'favorability') {
      value = Number(editFieldValue.value)
      if (isNaN(value) || value < 0 || value > 100) {
        notify('好感度需为 0-100 的数字', 'error')
        return
      }
    }
    if (editFieldType.value === 'group' && selectedGroupId.value) {
      await profileStore.updateGroupProfile(selectedGroupId.value, {
        [editFieldName.value]: value
      })
      notify('已更新')
    } else if (editFieldType.value === 'user' && selectedUserId.value) {
      await profileStore.updateUserProfile(selectedUserId.value, {
        [editFieldName.value]: value
      }, selectedUserGroupId.value)
      notify('已更新')
    }
  } catch (e: unknown) {
    notify(`更新失败: ${e instanceof Error ? e.message : '未知错误'}`, 'error')
  }
  showEditDialog.value = false
}

const startAddTag = (type: 'group' | 'user', field: string) => {
  addTagType.value = type
  addTagField.value = field
  addTagFieldLabel.value = FIELD_LABELS[field] || field
  addTagValue.value = ''
  showAddTagDialog.value = true
}

const submitAddTag = async () => {
  const val = addTagValue.value.trim()
  if (!val) return

  try {
    if (addTagType.value === 'group' && selectedGroupId.value) {
      const profile = profileStore.currentGroupProfile as GroupProfile | null
      const current = (profile as unknown as Record<string, string[] | undefined>)?.[addTagField.value] ?? []
      if (current.includes(val)) {
        notify('该项已存在', 'warning')
        return
      }
      await profileStore.updateGroupProfile(selectedGroupId.value, {
        [addTagField.value]: [...current, val]
      })
      notify('已添加')
    } else if (addTagType.value === 'user' && selectedUserId.value) {
      const profile = profileStore.currentUserProfile as UserProfile | null
      const current = (profile as unknown as Record<string, string[] | undefined>)?.[addTagField.value] ?? []
      if (current.includes(val)) {
        notify('该项已存在', 'warning')
        return
      }
      await profileStore.updateUserProfile(selectedUserId.value, {
        [addTagField.value]: [...current, val]
      }, selectedUserGroupId.value)
      notify('已添加')
    }
  } catch (e: unknown) {
    notify(`添加失败: ${e instanceof Error ? e.message : '未知错误'}`, 'error')
  }
  showAddTagDialog.value = false
}

const removeTagFromGroup = async (field: string, tag: string) => {
  if (!selectedGroupId.value || !profileStore.currentGroupProfile) return
  const current = (profileStore.currentGroupProfile as unknown as Record<string, string[] | undefined>)[field] ?? []
  const updated = current.filter(t => t !== tag)
  try {
    await profileStore.updateGroupProfile(selectedGroupId.value, {
      [field]: updated
    })
    notify('已移除')
  } catch (e: unknown) {
    notify(`移除失败: ${e instanceof Error ? e.message : '未知错误'}`, 'error')
  }
}

const removeTagFromUser = async (field: string, tag: string) => {
  if (!selectedUserId.value || !profileStore.currentUserProfile) return
  const current = (profileStore.currentUserProfile as unknown as Record<string, string[] | undefined>)[field] ?? []
  const updated = current.filter(t => t !== tag)
  try {
    await profileStore.updateUserProfile(selectedUserId.value, {
      [field]: updated
    }, selectedUserGroupId.value)
    notify('已移除')
  } catch (e: unknown) {
    notify(`移除失败: ${e instanceof Error ? e.message : '未知错误'}`, 'error')
  }
}

const removeDateFromUser = async (idx: number) => {
  if (!selectedUserId.value || !profileStore.currentUserProfile) return
  const current = profileStore.currentUserProfile.important_dates ?? []
  const updated = current.filter((_, i) => i !== idx)
  try {
    await profileStore.updateUserProfile(selectedUserId.value, {
      important_dates: updated
    }, selectedUserGroupId.value)
    notify('已移除')
  } catch (e: unknown) {
    notify(`移除失败: ${e instanceof Error ? e.message : '未知错误'}`, 'error')
  }
}

const submitAddDate = async () => {
  const date = addDateValue.value.trim()
  const desc = addDateDesc.value.trim()
  if (!date || !desc) return

  try {
    if (selectedUserId.value) {
      const current = profileStore.currentUserProfile?.important_dates ?? []
      await profileStore.updateUserProfile(selectedUserId.value, {
        important_dates: [...current, { date, description: desc }]
      }, selectedUserGroupId.value)
      notify('已添加')
    }
  } catch (e: unknown) {
    notify(`添加失败: ${e instanceof Error ? e.message : '未知错误'}`, 'error')
  }
  showAddDateDialog.value = false
}

const confirmDeleteGroup = () => {
  deleteDialogType.value = 'group'
  deleteDialogMessage.value = `确定要删除群聊「${profileStore.currentGroupProfile?.group_name || selectedGroupId.value}」的画像吗？此操作不可撤销。`
  showDeleteDialog.value = true
}

const confirmDeleteUser = () => {
  deleteDialogType.value = 'user'
  deleteDialogMessage.value = `确定要删除用户「${profileStore.currentUserProfile?.user_name || selectedUserId.value}」的画像吗？此操作不可撤销。`
  showDeleteDialog.value = true
}

const executeDelete = async () => {
  showDeleteDialog.value = false
  try {
    if (deleteDialogType.value === 'group' && selectedGroupId.value) {
      await profileStore.deleteGroupProfile(selectedGroupId.value)
      selectedGroupId.value = null
      notify('群聊画像已删除')
    } else if (deleteDialogType.value === 'user' && selectedUserId.value) {
      await profileStore.deleteUserProfile(selectedUserId.value, selectedUserGroupId.value)
      selectedUserId.value = null
      selectedUserGroupId.value = undefined
      // store 内部按删除目标的 groupId 刷新列表，可能与当前筛选器不一致，
      // 这里用当前筛选器重新加载以保证列表与筛选状态匹配
      loadUserList()
      notify('用户画像已删除')
    }
  } catch (e: unknown) {
    notify(`删除失败: ${e instanceof Error ? e.message : '未知错误'}`, 'error')
  }
}

const handleRefresh = () => {
  if (activeTab.value === 'group') {
    loadGroupList()
    if (selectedGroupId.value) loadGroupProfile()
  } else {
    loadUserList()
    if (selectedUserId.value) loadUserProfile()
  }
}

watch(activeTab, (newTab) => {
  if (newTab === 'group' && profileStore.groupList.length === 0) {
    loadGroupList()
  } else if (newTab === 'user' && profileStore.userList.length === 0) {
    loadUserList()
  }
})

// 群聊筛选变化时重新加载用户列表，并清空当前选中
watch(userGroupIdFilter, () => {
  selectedUserId.value = null
  selectedUserGroupId.value = undefined
  loadUserList()
})

onMounted(() => {
  loadGroupList()
  fetchGroups()
  window.addEventListener('iris:refresh', handleRefresh)
})
</script>

<style scoped>
.profile-view {
  height: 100%;
}

/* ProfileView 多了 v-window 包裹，flex 高度链断裂，
   需要显式给 v-card-text 设置 max-height 触发滚动 */
.profile-view :deep(.iris-list-card .v-card-text) {
  max-height: calc(100vh - 300px);
  overflow-y: auto;
  scrollbar-width: thin;
}

/* iris-list-card 已提供高度与滚动行为，这里保留局部细节 */

.profile-content {
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.profile-header {
  display: flex;
  align-items: center;
  padding-bottom: 16px;
  border-bottom: 1px solid rgba(var(--v-theme-on-surface), 0.08);
}

/* info-card 局部细节，hover 由 iris-card-hover 提供 */
.info-card {
  transition: box-shadow 0.2s ease;
}

.tags-container {
  display: flex;
  flex-wrap: wrap;
  margin: -4px;
}
</style>
