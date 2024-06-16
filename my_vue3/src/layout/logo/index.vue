<template>
    <div class="logo" v-if="setting.logoHidden">
        <img :src="setting.icon" alt="logo"/>
        <span class="title">{{setting.title}}</span>
    </div>
</template>

<script setup lang="ts">
import setting from '@/setting'
</script>

<style scoped lang="scss">
.logo {
    width: 95%;
    height: $base-menu-logo-height;
    display: flex;
    align-items: center;
    flex-wrap: nowrap;
    font-size: $base-logo-title-fontSize;
    font-weight: 700;
    color: #959ea6;
    padding-left: 17px;
    overflow: hidden;

    img {
        display: inline-block;
        width: 30px;
        height: 30px;
        border-radius: 50%;
    }

    .title {
        display: inline-block;
        margin: 2px 0px 0px 18px;
        overflow: hidden;
        font-size: 18px;
        font-weight: 600;
        line-height: 32px;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
}
</style>