<template>
    <el-card style="height: 450px">
        <template #header>
            <div class="card-header">
                <el-icon :size="20" style="vertical-align: middle;"><Monitor /></el-icon>
                <span>Main Menu</span>
            </div>
        </template>
        <el-menu
            :default-active="$route.path"
            active-text-color="#fff"
            text-color="#959ea6"
            :collapse="LayOutSettingStore.isCollapse"
            :router="true"
        >
            <template v-for="(item, index) in userStore.studyRoutes[0].children" :key="item.path">
                <el-menu-item v-if="!item.children" :index="item.path">
                    <def-svg-icon :name="item.meta.icon" color="$study-theme"></def-svg-icon>
                    <span style="margin-left: 10px">{{item.meta.title}}</span>
                </el-menu-item>
                <el-sub-menu v-if="item.children" :index="item.path">
                    <template #title>
                        <def-svg-icon :name="item.meta.icon" color="$study-theme"></def-svg-icon>
                        <span style="margin-left: 10px">{{item.meta.title + 'a'}}</span>
                    </template>
                    <template v-for="(item1, index1) in item.children" :key="item1.path">
                        <el-menu-item v-if="!item1.meta.hidden" :index="item1.path">
                            <def-svg-icon :name="item1.meta.icon" color="$study-theme"></def-svg-icon>
                            <span style="margin-left: 10px">{{item1.meta.title}}</span>
                        </el-menu-item>
                    </template>
                </el-sub-menu>
            </template>
        </el-menu>
    </el-card>
</template>

<script setup lang="ts">
import useLayOutSettingStore from '@/store/modules/setting'
import useUserStore from '@/store/modules/user'
import { useRoute, useRouter } from 'vue-router'

let $route = useRoute()
let $router = useRouter()
let userStore = useUserStore()  
let LayOutSettingStore = useLayOutSettingStore()

const backToHome = () => {
    $router.push({path: '/' || '/home'})
}
</script>

<style scoped lang="scss">
.el-card {
    /* padding: 20px; */
    transition: all 0.3s;
    margin:0 20px 20px 20px;
    border-radius: 10px;
    .card-header {
        font-size: 20px;
        font-weight: 700;
        margin-left: 10px;
        span {
            vertical-align: middle;
            margin-left: 10px;
        }
    }
}
.isCollapse {
  width: 56px;
}
.el-menu {
    border: none;
    .el-menu-item.is-active {
        font-weight: 700;
        background-color: #ff4b4b;
    }
    .el-menu-item {
        height: 35px;
        border-radius: 3px;
        padding-left: 10px !important;
        margin-bottom: 8px;
        transition: all 0.3s;
    }    
}
</style>