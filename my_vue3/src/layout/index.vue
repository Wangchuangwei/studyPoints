<template>
    <el-container class="layout-container-demo" style="height: 100vh">
        <el-aside
            :class="{isCollapse: LayOutSettingStore.isCollapse? true : false}"
        >
            <el-scrollbar>
                <el-menu
                    :default-active="$route.path"
                    active-text-color="#fff"
                    background-color="rgb(48, 65, 86)"
                    text-color="#959ea6"
                    :collapse="LayOutSettingStore.isCollapse"
                    :router="true"
                >
                    <Logo/>
                    <Menu :menuList="userStore.menuRoutes" />
                </el-menu>
            </el-scrollbar>
        </el-aside>
        <el-container>
            <TabBar />
            <el-main
                :style="{
                    left: !LayOutSettingStore.isCollapse ? '200px' : '56px',
                    width: LayOutSettingStore.isCollaps? 'calc(100% - 56px)' : 'calc(100% - 200px)'
                }"
            >
                <el-scrollbar>
                    <Main />
                </el-scrollbar>                
            </el-main>
        </el-container>
    </el-container>
</template>

<script setup lang="ts">
import Logo from './logo/index.vue'
import Menu from './menu/index.vue'
import TabBar from './tabbar/index.vue'
import Main from './main/index.vue'
import useLayOutSettingStore from '@/store/modules/setting'
import useUserStore from '@/store/modules/user'

import {useRoute} from 'vue-router'

const $route = useRoute()

// console.log($route, 'mathced', $route.matched)

let userStore = useUserStore()
let LayOutSettingStore = useLayOutSettingStore()
</script>

<style scoped lang="scss">
.layout-container-demo {
    height: 100%;
    .el-container {
        overflow: hidden;
    }
}

.layout-container-demo .el-menu {
    border-right: none;    
}

.layout-container-demo .el-main {
    position: absolute;
    left: $base-menu-width;
    top: $base-tabbar-height;
    padding: 10px;
    width: calc(100% - $base-menu-width);
    height: calc(100vh - $base-tabbar-height);
    transition: all 0.3s;
    .el-scrollbar .el-scrollbar__view {
        height: inherit !important;  
    }
}

.el-aside {
    width: $base-menu-width;
    background-color: rgb(48, 65, 86) !important;
    transition: all 0.3s;
}

.el-header {
    width:100%;
    background-color: #bbd9eb;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.12), 0 0 3px 0 rgba(0, 0, 0, 0.04);
    // z-index: 999;
}

.isCollapse {
    width: 56px;
}
</style>