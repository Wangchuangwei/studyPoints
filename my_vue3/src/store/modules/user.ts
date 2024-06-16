import { defineStore } from "pinia";
import router from '@/router'
import { constantRoute, asyncRoute, anyRoute, studyRoute } from "@/router/routes";
import type {UserState} from './types/types'
import {reqLogin, reqLogOut, reqUserInfo} from '@/api/user'
import type {LoginFormData, LoginResponseData, userInfoResponseData} from '@/api/type'
import {SET_TOKEN, GET_TOKEN, REMOVE_TOKEN} from '@/utils/token'

// @ts-ignore
import cloneDeep from 'lodash/cloneDeep'

function filterAsyncRoute(asyncRoute: any, routes: any) {
    return asyncRoute.filter((item: any) => {
      if (routes.includes(item.name)) {
        if (item.children && item.children.length > 0) {
          item.children = filterAsyncRoute(item.children, routes)
        }
        return true
      }
    })
  }

let useUserStore = defineStore('User', {
    state: (): UserState => {
        return {
            token: GET_TOKEN(),
            menuRoutes: constantRoute,
            studyRoutes: studyRoute,
            username: '',
            avatar: '',
            buttons: []
        }
    },
    actions: {
        async userLogin(data: LoginFormData) {
            let res: LoginResponseData = await reqLogin(data)
            if (res.code === 200) {
                this.token = res.data as string
                SET_TOKEN(res.data as string)
                return 'ok'
            } else{
                return Promise.reject(new Error(res.data as string))
            }
        },
        async userInfo() {
            let res: userInfoResponseData = await reqUserInfo()
            if (res.code === 200) {
                this.username = res.data.username as string
                this.avatar = res.data.avatar as string
                let userAsyncRoute = filterAsyncRoute(cloneDeep(asyncRoute), res.data.routes)
                this.menuRoutes = [...constantRoute, ...userAsyncRoute, anyRoute];
                [...userAsyncRoute, anyRoute].forEach((route: any) => {
                    router.addRoute(route)
                })
                return 'ok'
            } else {
                return Promise.reject(new Error(res.message))
            }
        },
        async userLogout() {
            let res = await reqLogOut()
            if (res.code === 200) {
                this.token = ''
                this.username = ''
                this.avatar = ''
                REMOVE_TOKEN()
            } else {
                return Promise.reject(new Error(res.message))
            }
        }
    },
    getters: {}
})

export default useUserStore