import { h, defineComponent } from 'vue'
import * as customIcons from './icons'

const mdiIconMap: Record<string, string> = {}
for (const [key, value] of Object.entries(customIcons)) {
  const kebab = key
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .toLowerCase()
  mdiIconMap[kebab] = value
}

export const MdiSvgIcon = defineComponent({
  name: 'MdiSvgIcon',
  inheritAttrs: false,
  props: {
    icon: {
      type: [String, Array, Object, Function],
      required: true
    },
    tag: {
      type: String,
      default: 'i'
    }
  },
  setup(props, { attrs }) {
    return () => {
      let icon = props.icon

      if (typeof icon === 'string') {
        if (icon.startsWith('mdi-')) {
          icon = mdiIconMap[icon] ?? icon
        }
      }

      return h(props.tag, { ...attrs, style: null }, {
        default: () => [
          h('svg', {
            class: 'v-icon__svg',
            xmlns: 'http://www.w3.org/2000/svg',
            viewBox: '0 0 24 24',
            role: 'img',
            'aria-hidden': 'true'
          }, [
            Array.isArray(icon)
              ? icon.map((p: any) =>
                  Array.isArray(p)
                    ? h('path', { d: p[0], 'fill-opacity': p[1] })
                    : h('path', { d: p })
                )
              : h('path', { d: icon })
          ])
        ]
      })
    }
  }
})
