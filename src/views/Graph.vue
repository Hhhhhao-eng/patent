<template>
  <div class="graph-container">
    <el-page-header @back="$router.go(-1)" title="返回" />
    
    <div class="graph-header">
      <h2>{{ patent.title }} - 知识图谱</h2>
      <div class="controls">
        <el-button icon="el-icon-refresh" @click="refreshGraph">重新布局</el-button>
        <el-button icon="el-icon-download" @click="exportImage">导出图片</el-button>
      </div>
    </div>
    
    <div ref="graph" class="graph-canvas"></div>
    
    <div v-if="loading" class="loading-overlay">
      <el-spinner size="large" />
      <p>正在生成知识图谱...</p>
    </div>
  </div>
</template>

<script>
import * as d3 from 'd3';
import patentApi from '@/api/patent';

export default {
  name: 'PatentGraph',
  props: ['id'],
  data() {
    return {
      patent: { title: '' },
      graphData: null,
      loading: false,
      svg: null,
      simulation: null
    }
  },
  mounted() {
    this.fetchGraphData();
  },
  methods: {
    async fetchGraphData() {
      try {
        this.loading = true;
        // 获取专利基础数据
        const patentRes = await patentApi.getPatentDetail(this.id);
        this.patent = patentRes.data;
        // 获取图谱数据
        const graphRes = await patentApi.getPatentGraph(this.id);
        this.graphData = graphRes.data;
        
        this.initGraph();
      } catch (err) {
        this.$message.error('获取图谱数据失败');
      } finally {
        this.loading = false;
      }
    },
    
    initGraph() {
      const container = this.$refs.graph;
      container.innerHTML = '';
      
      const width = container.clientWidth;
      const height = 600;
      
      // 创建SVG画布
      this.svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);
      
      // 创建力导向图
      this.simulation = d3.forceSimulation(this.graphData.nodes)
        .force('link', d3.forceLink(this.graphData.links).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2));
      
      // 绘制连线
      const link = this.svg.append('g')
        .selectAll('line')
        .data(this.graphData.links)
        .enter()
        .append('line')
        .attr('stroke', '#99a9bf')
        .attr('stroke-width', 2);
      
      // 绘制节点
      const node = this.svg.append('g')
        .selectAll('circle')
        .data(this.graphData.nodes)
        .enter()
        .append('circle')
        .attr('r', d => d.type === 'patent' ? 10 : 8)
        .attr('fill', d => d.type === 'patent' ? '#409EFF' : '#67C23A')
        .call(d3.drag()
          .on('start', this.dragStarted)
          .on('drag', this.dragged)
          .on('end', this.dragEnded));
      
      // 添加标签
      const text = this.svg.append('g')
        .selectAll('text')
        .data(this.graphData.nodes)
        .enter()
        .append('text')
        .text(d => d.name)
        .attr('font-size', 12)
        .attr('dx', 15)
        .attr('dy', 4);
      
      // 更新位置
      this.simulation.on('tick', () => {
        link
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);
          
        node
          .attr('cx', d => d.x)
          .attr('cy', d => d.y);
          
        text
          .attr('x', d => d.x)
          .attr('y', d => d.y);
      });
    },
    
    dragStarted(event, d) {
      if (!event.active) this.simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    },
    
    dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    },
    
    dragEnded(event, d) {
      if (!event.active) this.simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    },
    
    refreshGraph() {
      this.simulation.alpha(1).restart();
    },
    
    exportImage() {
      const svgData = new XMLSerializer().serializeToString(this.svg.node());
      const blob = new Blob([svgData], { type: 'image/svg+xml' });
      const url = URL.createObjectURL(blob);
      
      const link = document.createElement('a');
      link.href = url;
      link.download = `patent-graph-${this.id}.svg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }
}
</script>

<style scoped>
.graph-container {
  max-width: 1200px;
  margin: 20px auto;
  padding: 20px;
}

.graph-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 20px 0;
}

.graph-canvas {
  width: 100%;
  height: 600px;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  background-color: #fafafa;
  position: relative;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.8);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  z-index: 10;
}
</style>