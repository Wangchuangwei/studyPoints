const diffsRouter = {
    path: '/diffs',
    component: () => import('@/layout/index.vue'),
    name: 'Diffs',
    meta: {
        hidden: false,
        title: '差异比对',
        icon: 'Platform'
    },
    children: [
        {
            path: '/diffs/filediff',
            component: () => import('@/views/profile/index.vue'),
            name: 'FileDiff',
            meta: {
                title: '文件级差异',
                hidden: false,
                icon: 'Platform'
            }
        },
        {
            path: '/diffs/linediff',
            component: () => import('@/views/profile/index.vue'),
            name: 'LineDiff',
            meta: {
                title: '行级差异',
                hidden: false,
                icon: 'Platform'
            }
        }
    ]
}

export default diffsRouter