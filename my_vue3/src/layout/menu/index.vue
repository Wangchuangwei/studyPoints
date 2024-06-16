<script setup lang="ts" name="Menu">
defineProps(['menuList'])
</script>
<template>
  <template v-for="(item, index) in menuList" :key="item.path">
      <div v-if="!item.meta.hidden">
        <!-- 没有子路由 -->
        <template v-if="!item.children">
            <el-menu-item :index="item.path">
                <el-icon>
                    <component :is="item.meta.icon"></component>
                </el-icon>
                <span>{{item.meta.title}}</span>
            </el-menu-item>
        </template>
        <!-- 只有一个子路由 -->
        <template v-if="item.children && item.children.length === 1">
            <el-menu-item v-if="!item.children[0].meta.hidden" :index="item.children[0].path">
                <el-icon>
                    <component :is="item.children[0].meta.icon"></component>
                </el-icon>
                <span>{{item.children[0].meta.title}}</span>
            </el-menu-item>
        </template>
        <!-- 含有多个子路由 -->
        <el-sub-menu v-if="item.children && item.children.length > 1" :index="item.path">
            <template #title>
                <el-icon>
                    <component :is="item.meta.icon"></component>
                </el-icon>
                <span>{{item.meta.title}}</span>
            </template>
            <template v-for="(item1, index1) in item.children" :key="item1.path">
                <el-menu-item v-if="!item1.meta.hidden" :index="item1.path">
                    <el-icon>
                        <component :is="item1.meta.icon"></component>
                    </el-icon>
                    <span>{{item1.meta.title}}</span>
                </el-menu-item>
            </template>
        </el-sub-menu>
      </div>
  </template>
</template>
<style scoped lang="scss">
span{
    margin-left: 7px;
}
</style>
