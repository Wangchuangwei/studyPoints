<template>
    <el-container class="container">
        <el-aside class="layout-container-aside">
            <!-- <div class="fixed-button" @click="backToHome" >
                <el-icon style="vertical-align: middle;" size="30"><Back /></el-icon>
                <span>返回首页</span>
            </div> -->
            <left></left> 
        </el-aside>
        <el-main class="layout-container-main">
            <router-view></router-view>
        </el-main>
    </el-container>
</template>

<script setup lang="ts">
import {ref, onMounted} from 'vue'
import Left from './components/left/index.vue'

import { useRoute, useRouter } from 'vue-router'

let $route = useRoute()
let $router = useRouter()
onMounted(() => {
  console.log('saaa:', $route.path) 
})

const backToHome = () => {
    $router.push({path: '/home'})
}
</script>

<style scoped lang="scss">
.container {
    width: 100vw;
    height: 100vh;
    .layout-container-aside {
        background: #f0f2f6;
        display: flex;
        flex-direction: column;
        justify-content: center;
        // padding-left: 20px;
    }
}
.fixed-button {
    position: absolute;
    top: 10px;
    left: 20px;
    padding: 7px;
    span {
        vertical-align: middle;
    }
    &:hover {
        border: 1px solid #e2e2e2;
        border-radius: 3px;
        background-color: #464646;
        color: #f1f3f4;
    }
}
</style>
<style lang="scss">
.title {
    font-size: 20px; 
    font-weight: 700;
    margin-bottom: 20px;
    color: #4c5e7d;
}
.setting-title {
    font-weight: 300;
    margin-bottom: 10px;
}
.show_result {
    height: 40px;
    width: 500px;
    background: #e8f2fc;
    line-height: 40px;
    padding-left: 20px;
    color: #5975a2;
    margin-bottom: 20px;
}
.el-select {
    .el-input.is-focus .el-input__wrapper {
        box-shadow: 0 0 0 1px $study-theme !important;
    }
    .el-input__wrapper.is-focus {
        box-shadow: 0 0 0 1px $study-theme !important;
    }    
}
.el-input__wrapper {
    background-color: #f0f2f6;
}
.el-input__wrapper.is-focus {
    box-shadow: 0 0 0 1px $study-theme;
}
</style>