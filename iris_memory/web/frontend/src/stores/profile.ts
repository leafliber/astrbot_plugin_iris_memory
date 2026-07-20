import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { GroupProfile, UserProfile, GroupListItem, UserListItem } from '@/types'
import * as profileApi from '@/api/profile'

export const useProfileStore = defineStore('profile', () => {
  const groupList = ref<GroupListItem[]>([])
  const groupListLoading = ref(false)

  const currentGroupProfile = ref<GroupProfile | null>(null)
  const groupProfileLoading = ref(false)

  const userList = ref<UserListItem[]>([])
  const userListLoading = ref(false)

  const currentUserProfile = ref<UserProfile | null>(null)
  const userProfileLoading = ref(false)

  const fetchGroupList = async () => {
    groupListLoading.value = true
    try {
      groupList.value = await profileApi.getGroupList()
    } catch (error) {
      console.error('获取群聊列表失败:', error)
      groupList.value = []
    } finally {
      groupListLoading.value = false
    }
  }

  const fetchGroupProfile = async (groupId: string) => {
    groupProfileLoading.value = true
    try {
      currentGroupProfile.value = await profileApi.getGroupProfile(groupId)
    } catch (error) {
      console.error('获取群聊画像失败:', error)
      currentGroupProfile.value = null
    } finally {
      groupProfileLoading.value = false
    }
  }

  const updateGroupProfile = async (groupId: string, data: Partial<GroupProfile>) => {
    await profileApi.updateGroupProfile(groupId, data)
    await fetchGroupProfile(groupId)
  }

  const fetchUserList = async (groupId?: string) => {
    userListLoading.value = true
    try {
      userList.value = await profileApi.getUserList(groupId)
    } catch (error) {
      console.error('获取用户列表失败:', error)
      userList.value = []
    } finally {
      userListLoading.value = false
    }
  }

  const fetchUserProfile = async (userId: string, groupId?: string) => {
    userProfileLoading.value = true
    try {
      currentUserProfile.value = await profileApi.getUserProfile(userId, groupId)
    } catch (error) {
      console.error('获取用户画像失败:', error)
      currentUserProfile.value = null
    } finally {
      userProfileLoading.value = false
    }
  }

  const updateUserProfile = async (userId: string, data: Partial<UserProfile>, groupId?: string) => {
    await profileApi.updateUserProfile(userId, data, groupId)
    await fetchUserProfile(userId, groupId)
  }

  const deleteGroupProfile = async (groupId: string) => {
    await profileApi.deleteGroupProfile(groupId)
    currentGroupProfile.value = null
    await fetchGroupList()
  }

  const deleteUserProfile = async (userId: string, groupId?: string) => {
    await profileApi.deleteUserProfile(userId, groupId)
    currentUserProfile.value = null
    await fetchUserList(groupId)
  }

  return {
    groupList,
    groupListLoading,
    currentGroupProfile,
    groupProfileLoading,
    userList,
    userListLoading,
    currentUserProfile,
    userProfileLoading,
    fetchGroupList,
    fetchGroupProfile,
    updateGroupProfile,
    fetchUserList,
    fetchUserProfile,
    updateUserProfile,
    deleteGroupProfile,
    deleteUserProfile
  }
})
