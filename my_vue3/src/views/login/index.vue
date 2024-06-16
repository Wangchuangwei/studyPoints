<template>
    <div class="login_container">
        <div class="center">
            <el-form :model="loginForm" :rules="rules" class="login_form" ref="loginForms">
                <h1>测试</h1>
                <el-form-item prop="username">
                    <el-input
                        :prefix-icon="User"
                        v-model="loginForm.username"
                        placeholder="Username"
                        size="large"
                    ></el-input>
                </el-form-item>
                <el-form-item prop="password">
                    <el-input
                        type="password"
                        show-password
                        :prefix-icon="Lock"
                        v-model="loginForm.password"
                        placeholder="Password"
                        size="large"
                        clearable
                    ></el-input>
                </el-form-item>
                <el-form-item prop="verifyCode">
                    <el-input
                        show-password
                        :prefix-icon="Warning"
                        v-model="loginForm.verifyCode"
                        placeholder="VerifyCode"
                        size="large"
                        maxlength="4"
                    >
                        <template #append>
                            <Identify :identifyCode="identifyCode" @click="refreshCode"/>
                        </template>
                    </el-input>
                </el-form-item>
                <el-form-item>
                    <el-button
                        :loading="loading"
                        type="primary"
                        size="default"
                        class="login_btn"
                        @click="login"
                    >登录</el-button>
                </el-form-item>
            </el-form>  
        </div>      
    </div>
</template>

<script setup lang="ts">
import {User, Lock, Warning} from '@element-plus/icons-vue'
import { ElNotification } from 'element-plus'
import {Ref, ref, reactive} from 'vue'
import {useRouter, useRoute} from 'vue-router'
import router from '@/router'
import useUserStore from '@/store/modules/user'
import {getTime} from '@/utils/time'
import Identify from '@/components/VerifyCode/index.vue' 

let $router = useRouter()
let $route = useRoute()

let loginForms = ref()
let loading = ref(false)
let useStore = useUserStore()

/**验证码*/
const identifyCode = ref('1234')
const identifyCodes = ref('1234567890abcdefjhijklinopqrsduvwxyz')

const loginForm = reactive({
    username: 'admin',
    password: '123456',
    verifyCode: '1234'
})

const validatorUsername = (rule: any, value: any, callback: any) => {
    if (value.length === 0){
        callback(new Error('请输入账号'))
    } else {
        callback()
    }
}

const validatorPassword = (rule: any, value: any, callback: any) => {
    if (value.length === 0) {
        callback(new Error('请输入密码'))
    } else if (value.length < 6 || value.length > 16) {
        callback(new Error('密码应为6~16位的任意组合'))
    } else {
        callback()
    }
}

const validatorVerifyCode = (rule: any, value: any, callback: any) => {
    if (value.length === 0) {
        callback(new Error('请输入验证码'))
    } else if (value.length < 4) {
        callback(new Error('请输入正确的验证码'))
    } else if (value !== identifyCode.value) {
        callback(new Error('请输入正确的验证码'))
    } else if (value === identifyCode.value) {
        callback()
    }
}

const rules = {
    username: [
        { trigger: 'change', validator: validatorUsername},
    ],
    password: [
        { trigger: 'change', validator: validatorPassword},   
    ],
    verifyCode: [
        { trigger: 'blur', validator: validatorVerifyCode}    
    ]
}

const login = async () => {
    await loginForms.value.validate()
    // let request1 = axios.create({
    //     // baseURL: import.meta.env.VITE_APP_BASE_API,
    //     timeout: 5000
    // })
    // request1.post(
    //     '/api/user/login',
    //     {
    //         username: 'admin',
    //         password: '123456',
    //     }   
    // ).then(res => console.log('从高：', res, res.data))
    // .catch(error => console.log('失败', error))

    loading.value = true
    try {
        await useStore.userLogin(loginForm)
        let redirect: string = $route.query.redirect as string
        $router.push({path: redirect || '/'})
        // $router.push({path: '/'})
        console.log('router-----:', $router)
        ElNotification({
            type: 'success',
            message: '登录成功',
            title: `Hi, ${getTime()}好`
        })
        loading.value = false
    } catch (error) {
        loading.value = false
        ElNotification({
            type: 'error',
            message: (error as Error).message,
        })
    }
}

const refreshCode = () => {
    identifyCode.value = ''
    makeCode(identifyCode, 4)
}

const makeCode = (o: Ref<string>,l: number) => {
    for (let i = 0; i < l; i++) {
        o.value += identifyCodes.value[randomNum(0, identifyCodes.value.length)]
    }
}

const randomNum = (min: number, max: number) => {
    return Math.floor(Math.random() * (max - min) + min)
}

</script>

<style scoped lang="scss">
.login_container {
    height: 100vh;
    width: 100%;
    background: linear-gradient(to left top, #5f98cd, #6ac2e3);
    overflow: hidden;

    .center {
        position: absolute;
        width: 400px;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: transparent;
        box-shadow: 1px 2px 10px 0 rgba(0,0,0,0.3);
        overflow: hidden;

        .login_form {
            width: 100%;
            padding: 10px;

            h1 {
                background: linear-gradient(to right, #fbfbfb, #4cb6de);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                font-size: 40px;
                font-weight: 700;
                margin-bottom: 40px;
                margin-top: 10px;
            }

            .login_btn {
                width: 100%;
            }
        }
    }
}
:deep(.el-input-group__append, .el-input-group__prepend) {
    padding: 0;
}
</style>