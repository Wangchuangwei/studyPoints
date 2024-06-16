<template>
    <el-row :gutter="20">
        <el-col :span="6" :xs="24"><UserCard/></el-col>
        <el-col :span="18" :xs="24">
            <el-card>
                <el-tabs v-model="activeTab">
                    <el-tab-pane label="Activity" name="activity">a</el-tab-pane>
                    <el-tab-pane label="Timeline" name="timeline"><Timeline/></el-tab-pane>
                    <el-tab-pane label="Account" name="account"><Account/></el-tab-pane>
                </el-tabs>
            </el-card>
        </el-col>
    </el-row>
</template>

<script setup lang="ts">
import {ref} from 'vue'
import UserCard from './components/UserCard.vue'
import Account from './components/Account.vue'
import Timeline from './components/Timeline.vue'

let activeTab = ref<string>('timeline')
</script>

<style scoped>

</style>