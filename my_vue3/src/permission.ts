import router from '@/router'
import setting from './setting'
import pinia from './store'
import useUserStore from './store/modules/user'
import nprogress from 'nprogress'
import 'nprogress/nprogress.css'

let userStore = useUserStore(pinia)
router.beforeEach(async (to, from, next) => {
    document.title = to.meta.title + ` | ${setting.title}`
    nprogress.start()
    let token = userStore.token
    let username = userStore.username
    console.log('进入授权')
    if (token) {
        if (to.path == '/login') {
            next({path: '/'})
        } else {
            if (username) {
                next()
            } else {
                try {
                    await userStore.userInfo()
                    next()
                } catch (error) {
                    await userStore.userLogout()
                    next({path: '/login', query: {redirect: to.path}})
                }
            }
        }
    } else {
        if (to.path === '/login') {
            next()
        } else {
            console.log('没有授权')
            next({path: '/login', query: {redirect: to.path}})
        }
    }
})

router.afterEach((route) => {
    nprogress.done()
})