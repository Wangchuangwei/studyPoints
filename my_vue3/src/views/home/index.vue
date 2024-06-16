<template>
    <el-card>
        <div class="box">
            <img :src="setting.logo" alt="" class="logo" />
            <div class="footer">
                <h3 class="title">
                    {{getTime()}}好~
                    <span class="gradient">{{userStore.username}}</span>
                </h3>
                <p class="subtitle">原型系统</p>
            </div>
        </div>
    </el-card>
</template>

<script setup lang="ts">
import setting from '@/setting'
import {getTime} from '@/utils/time'
import useUserStore from '@/store/modules/user'
import {useRoute} from 'vue-router'

let userStore = useUserStore()
</script>

<style scoped lang='scss'>
.box {
    display: flex;
    .logo {
        width: 100px;
        height: 100px;
        border-radius: 50%;
    }
    .footer {
        margin-left: 20px;
        margin-top: 15px;
        .title {
            font-size: 35px;
            font-weight: 900;
            margin-bottom: 30px;
            .gradient {
                background: linear-gradient(to right, #001529, #fff);
                background-clip: text;
                -webkit-background-clip: text;
                color: transparent;
                font-size: 24px;
                font-weight: bold;
            }
        }
        .subtitle {
            color: #cccccc;
            font-style: italic;
            font-weight: 700;
        }
    }

}
</style>