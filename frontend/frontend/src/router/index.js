import { createRouter, createWebHistory } from 'vue-router'
import KGPC from '@/views/AI/KGPC/index.vue'
import PatentDetail from '@/views/Detail.vue'
import PatentGraph from '@/views/Graph.vue'
import RelationInfer from '@/views/AI/KGPC/RelationInfer.vue'
import PathFinder from '@/views/AI/KGPC/PathFinder.vue'

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
    path: '/AI/KGPC/relation-infer',
    name: 'RelationInfer',
    component: RelationInfer
  },
  {
    path: '/AI/KGPC/path-finder',
    name: 'PathFinder',
    component: PathFinder
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