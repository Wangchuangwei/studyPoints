const downloadsRoutes = {
    path: '/downloads',
    component: () => import('@/layout/index.vue'),
    name: 'Downloads',
    meta: {
        hidden: false,
        title: '导出文件',
        icon: 'Platform'
    },
    children: [
        {
            path: '/downloads/studyfile',
            component: () => import('@/views/export/StudyFile.vue'),
            name: 'StudyFile',
            meta: {
                title: '研究文件',
                hidden: false,
                icon: 'Platform'
            }
         },
        {
            path: '/downloads/userfile',
            component: () => import('@/views/export/UserFile.vue'),
            name: 'UserFile',
            meta: {
                title: '用户文件',
                hidden: false,
                icon: 'Platform'
            }
         },
    ]
}

export default downloadsRoutes