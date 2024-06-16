import request from '@/utils/request'
import type {LoginFormData, LoginResponseData, userInfoResponseData} from '../type/index'

enum API {
    LOGIN_URL = '/api/user/login',
    USERINFRO_URL = '/api/user/info',
    LOGOUT_URL = '/api/user/logout'
}

export const reqLogin = (data: LoginFormData) => request.post<any, LoginResponseData>(API.LOGIN_URL, data) 

export const reqUserInfo = () => request.get<any, userInfoResponseData>(API.USERINFRO_URL)

export const reqLogOut = () => request.post<any, any>(API.LOGOUT_URL)
