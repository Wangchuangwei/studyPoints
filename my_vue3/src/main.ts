import { createApp } from 'vue'
import App from '@/App.vue'
import 'virtual:svg-icons-register'

import globalComponent from '@/components'
// import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/el-notification.css'
import 'element-plus/theme-chalk/dark/css-vars.css'
import '@/styles/index.scss'
import router from './router'
import pinia from './store'
import './permission'

const app = createApp(App)
app.use(globalComponent)
app.use(router)
app.use(pinia)
app.mount('#app')

