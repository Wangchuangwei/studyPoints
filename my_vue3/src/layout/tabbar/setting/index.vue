<template>
    <el-button circle size="small" :icon="Refresh" @click="updateRefresh"></el-button>
    <el-button circle size="small" :icon="FullScreen" @click="fullScreen"></el-button>
    <el-popover placement="bottom" title="主题设置" :width="200" trigger="hover">
        <el-form>
            <el-form-item label="主题颜色">
                <el-color-picker 
                    v-model="color"
                    show-alpha
                    :predefine="predefineColors"
                    size="small"
                    @change="setColor"
                />
            </el-form-item>
            <el-form-item label="暗黑模式">
                <el-switch
                    v-model="dark"
                    size="small"
                    inline-prompt
                    active-icon="MoonNight"
                    inactive-icon="Sunny"
                    @change="changeDark"
                />
            </el-form-item>
        </el-form>
        <template #reference>
            <el-button circle size="small" :icon="Setting" />
        </template>
    </el-popover> 
    <img :src="setting.logo" alt="" />
    <el-dropdown>
        <span class="el-dropdown-link" style="cursor: pointer">
            {{userStore.username}}
            <el-icon class="el-icon--right"><arrow-down /></el-icon>
        </span>
        <template #dropdown>
            <el-dropdown-menu>
                <router-link to="/profile/index">
                    <el-dropdown-item>系统简介</el-dropdown-item>
                </router-link>
                <a target="_blank" href="https://element-plus.gitee.io/zh-CN/guide/installation.html">
                    <el-dropdown-item>UI文档</el-dropdown-item>
                </a>
                <el-dropdown-item divided @click="logout">退出登录</el-dropdown-item>
            </el-dropdown-menu>
        </template>
    </el-dropdown>
</template>

<script setup lang="ts">
import {ref} from 'vue'
import {Refresh, Setting, FullScreen, ArrowDown} from '@element-plus/icons-vue'
import {useRouter, useRoute} from 'vue-router'
import useLayOutSettingStore from '@/store/modules/setting'
import useUserStore from '@/store/modules/user'
import setting from '@/setting'

let $router = useRouter()
let $route = useRoute()
let userStore = useUserStore()
let LayOutSettingStore = useLayOutSettingStore()

const updateRefresh = () => {
    LayOutSettingStore.refsh = !LayOutSettingStore.refsh
}

const fullScreen = () => {
    let full = document.fullscreenElement
    if (!full) {
        document.documentElement.requestFullscreen()
    } else {
        document.exitFullscreen()
    }
}

let color = ref('rgba(255, 69, 0, 0.68)')
const predefineColors = ref([
    '#ff4500',
    '#ff8c00',
    '#ffd700',
    '#90ee90',
    '#00ced1',
    '#1e90ff',
    '#c71585',
    'rgba(255, 69, 0, 0.68)',
    'rgb(255, 120, 0)',
    'hsv(51, 100, 98)',
    'hsva(120, 40, 94, 0.5)',
    'hsl(181, 100%, 37%)',
    'hsla(209, 100%, 56%, 0.73)',
    '#c7158577',
])

const setColor = () => {
    let html = document.documentElement
    html.style.setProperty('--el-color-primary', color.value)
}

let dark = ref<boolean>(false)

const changeDark = () => {
    let html = document.documentElement
    dark.value? (html.className = 'dark') : (html.className = '')
}

const logout = async() => {
    await userStore.userLogout()
    $router.push({path: '/login', query: {redirect: $route.path}})
}

</script>

<style scoped lang="scss">
img {
    width: 24px;
    height: 24px;
    border-radius: 20px;
    margin: 0 10px; 
}  
</style>