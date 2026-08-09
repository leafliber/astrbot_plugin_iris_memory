<template>
  <div class="grapheme-diff">
    <div class="diff-toolbar d-flex align-center flex-wrap ga-2 mb-2">
      <v-chip size="small" color="success" variant="tonal">+{{ summary.inserted }}</v-chip>
      <v-chip size="small" color="error" variant="tonal">−{{ summary.deleted }}</v-chip>
      <v-chip size="small" variant="tonal">未变 {{ summary.unchanged }}</v-chip>
      <v-spacer />
      <v-switch
        v-model="showUnchanged"
        label="显示未修改内容"
        color="primary"
        density="compact"
        hide-details
      />
    </div>
    <div class="diff-content" role="region" aria-label="人格版本字符差异">
      <template v-for="(chunk, index) in visibleChunks" :key="index">
        <span
          v-if="chunk.kind !== 'equal' || showUnchanged"
          :class="`diff-${chunk.kind}`"
        >{{ chunk.text }}</span>
        <span v-else class="diff-gap" :title="`折叠 ${chunk.graphemeCount} 个未修改字符`">
          …
        </span>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { diffGraphemes, summarizeDiff } from '@/utils/graphemeDiff'

const props = defineProps<{
  before: string
  after: string
}>()

const showUnchanged = ref(true)
const chunks = computed(() => diffGraphemes(props.before || '', props.after || ''))
const summary = computed(() => summarizeDiff(chunks.value))
const visibleChunks = computed(() => chunks.value)
</script>

<style scoped>
.diff-content {
  max-height: min(54vh, 680px);
  overflow: auto;
  padding: 16px;
  border: 1px solid rgba(var(--v-theme-on-surface), 0.12);
  border-radius: 10px;
  background: rgba(var(--v-theme-on-surface), 0.025);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.86rem;
  line-height: 1.75;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.diff-insert,
.diff-delete {
  border-radius: 3px;
  padding: 1px 0;
}

.diff-insert {
  color: rgb(var(--v-theme-success));
  background: rgba(var(--v-theme-success), 0.18);
  text-decoration: none;
}

.diff-delete {
  color: rgb(var(--v-theme-error));
  background: rgba(var(--v-theme-error), 0.16);
  text-decoration: line-through;
  text-decoration-thickness: 1px;
}

.diff-gap {
  display: inline-block;
  margin: 0 4px;
  padding: 0 6px;
  color: rgba(var(--v-theme-on-surface), 0.5);
  border-radius: 4px;
  background: rgba(var(--v-theme-on-surface), 0.06);
}

.diff-toolbar :deep(.v-switch) {
  flex: 0 0 auto;
}
</style>
