<template>
    <el-form :model="accountForm" label-position="top">
        <el-form-item label="Name">
            <el-input v-model.trim="accountForm.username"></el-input>
        </el-form-item>
        <el-form-item label="Email">
            <el-input v-model.trim="accountForm.email"></el-input>
        </el-form-item>
        <el-form-item>
            <el-button
                type="primary"
                size="default"
                @click="submit"
            >Update</el-button>
        </el-form-item>
    </el-form>
</template>

<script setup lang="ts" name="Account">
import {reactive} from 'vue'
import {AboutAuthor} from '@/store/modules/setting'

let aboutAuthor = AboutAuthor()

const accountForm = reactive({
    username: aboutAuthor.username|| 'admin',
    email: aboutAuthor.email || '1032273697@qq.com'
})

const submit = () => {
    aboutAuthor.authorInfo()
}
</script>

<style scoped lang="scss">

</style>