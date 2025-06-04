import { createRouter, createWebHistory } from 'vue-router'
import KGPC from '@/views/AI/KGPC/index.vue'
import PatentDetail from '@/views/Detail.vue'
import PatentGraph from '@/views/Graph.vue'

const routes = [
  {
    path: '/AI/KGPC',
    name: 'KGPC',
    component: KGPC
  },
  {
    path: '/AI/KGPC/detail/:id',
    name: 'PatentDetail',
    component: PatentDetail,
    props: true
  },
  {
    path: '/AI/KGPC/graph/:id',
    name: 'PatentGraph',
    component: PatentGraph,
    props: true
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: '/AI/KGPC'
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router