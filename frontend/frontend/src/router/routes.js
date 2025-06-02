// 专利知识图谱页面组件
const PatentKG = () => import('@/views/AI/KGPC/index.vue')

export default [
  {
    path: '/AI/KGPC',
    name: 'PatentKG',
    component: PatentKG,
    meta: {
      title: '专利知识图谱',
      requiresAuth: true // 需要登录验证
    }
  },
  // 其他路由可以在这里添加
  {
    path: '*',
    redirect: '/AI/KGPC' // 默认重定向到专利页面
  }
]