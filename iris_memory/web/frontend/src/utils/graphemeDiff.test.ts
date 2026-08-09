import { describe, expect, it } from 'vitest'
import {
  diffGraphemes,
  fallbackSegmentGraphemes,
  segmentGraphemes,
  summarizeDiff
} from './graphemeDiff'

describe('segmentGraphemes', () => {
  it('keeps Chinese, newlines and ASCII as user-perceived characters', () => {
    expect(segmentGraphemes('你好\nIris')).toEqual(['你', '好', '\n', 'I', 'r', 'i', 's'])
  })

  it('does not split emoji, ZWJ emoji, flags or combining marks', () => {
    const text = '👍🏽👨‍👩‍👧‍👦🇨🇳e\u0301'
    expect(segmentGraphemes(text)).toEqual(['👍🏽', '👨‍👩‍👧‍👦', '🇨🇳', 'e\u0301'])
  })

  it('provides a safe fallback for common extended grapheme clusters', () => {
    expect(fallbackSegmentGraphemes('👍🏽👩‍💻🇨🇳e\u0301')).toEqual(['👍🏽', '👩‍💻', '🇨🇳', 'e\u0301'])
  })
})

describe('diffGraphemes', () => {
  it('returns a shortest readable edit sequence for Chinese text', () => {
    const chunks = diffGraphemes('语气自然。', '语气更自然！')
    expect(chunks.map(chunk => [chunk.kind, chunk.text])).toEqual([
      ['equal', '语气'],
      ['insert', '更'],
      ['equal', '自然'],
      ['delete', '。'],
      ['insert', '！']
    ])
    expect(summarizeDiff(chunks)).toEqual({ inserted: 2, deleted: 1, unchanged: 4 })
  })

  it('treats a ZWJ emoji replacement as one deletion and one insertion', () => {
    const chunks = diffGraphemes('开发者 👩‍💻', '开发者 👨‍💻')
    expect(summarizeDiff(chunks)).toEqual({ inserted: 1, deleted: 1, unchanged: 4 })
    expect(chunks.find(chunk => chunk.kind === 'delete')?.text).toBe('👩‍💻')
    expect(chunks.find(chunk => chunk.kind === 'insert')?.text).toBe('👨‍💻')
  })

  it('does not split combining characters in fallback mode', () => {
    const chunks = diffGraphemes('cafe\u0301', 'café', true)
    expect(summarizeDiff(chunks)).toEqual({ inserted: 1, deleted: 1, unchanged: 3 })
    expect(chunks.find(chunk => chunk.kind === 'delete')?.text).toBe('e\u0301')
  })

  it('handles empty and identical snapshots', () => {
    expect(diffGraphemes('', '人格')).toEqual([{ kind: 'insert', text: '人格', graphemeCount: 2 }])
    expect(diffGraphemes('人格', '')).toEqual([{ kind: 'delete', text: '人格', graphemeCount: 2 }])
    expect(diffGraphemes('人格', '人格')).toEqual([{ kind: 'equal', text: '人格', graphemeCount: 2 }])
  })
})
