<template>
  <div class="graph-container">
    <div class="card">
      <el-page-header @back="$router.go(-1)" title="返回" />
      <div class="graph-header">
        <h2>{{ patent.title }} <span class="subtitle">知识图谱</span></h2>
        <div class="controls">
          <el-button icon="el-icon-download" @click="exportImage">导出图片</el-button>
        </div>
      </div>
      <div ref="graph" class="graph-canvas"></div>
      <div v-if="loading" class="loading-overlay">
        <el-spinner size="large" />
        <p>正在生成知识图谱...</p>
      </div>
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
      simulation: null,
      exporting: false,
      depth: undefined, // 不再使用
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
        this.patent = patentRes.data.data;
        // 获取图谱数据（不再传递depth参数）
        const graphRes = await patentApi.getPatentGraph(this.id);
        this.graphData = graphRes.data.data;
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
      if (this.graphData && this.graphData.nodes && this.graphData.links) {
        const nodeIds = new Set(this.graphData.nodes.map(n => n.id));
        this.graphData.links = this.graphData.links.filter(
          l => nodeIds.has(l.source) && nodeIds.has(l.target)
        );
      }
      if (!this.graphData || !this.graphData.nodes || !this.graphData.links) {
        this.$message.error('知识图谱数据为空');
        return;
      }
      if (this.graphData.nodes.length === 1 && this.graphData.links.length === 0) {
        container.innerHTML = '<div style="text-align:center;color:#999;padding:60px 0;">该专利暂无关联关系，仅有自身节点</div>';
        return;
      }
      const width = container.clientWidth || 900;
      const height = 600;
      container.style.resize = 'both';
      container.style.overflow = 'auto';

      // 创建SVG画布，支持缩放和平移
      this.svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height)
        .attr('viewBox', [0, 0, width, height].join(' '))
        .call(
          d3.zoom()
            .scaleExtent([0.2, 5])
            .on('zoom', (event) => {
              g.attr('transform', event.transform);
            })
        );

      // 创建分组用于缩放
      const g = this.svg.append('g');

      // 创建力导向图
      this.simulation = d3.forceSimulation(this.graphData.nodes)
        .force('link', d3.forceLink(this.graphData.links).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2));

      // 绘制连线
      const link = g.append('g')
        .selectAll('line')
        .data(this.graphData.links)
        .enter()
        .append('line')
        .attr('stroke', '#99a9bf')
        .attr('stroke-width', 2);

      // 绘制节点
      const node = g.append('g')
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
      const text = g.append('g')
        .selectAll('text')
        .data(this.graphData.nodes)
        .enter()
        .append('text')
        .text(d => d.label || d.name || d.id)
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
    },
    
    async exportToNeo4j() {
      if (this.exporting) return;
      this.exporting = true;
      try {
        const res = await patentApi.exportGraphToNeo4j(this.id);
        if (res.data && res.data.success) {
          this.$message.success('导出到Neo4j成功！');
        } else {
          this.$message.error(res.data && res.data.error ? res.data.error : '导出失败');
        }
      } catch (err) {
        this.$message.error('导出到Neo4j失败');
      } finally {
        this.exporting = false;
      }
    }
  }
}
</script>

<style scoped>
.graph-container {
  max-width: 1200px;
  margin: 40px auto;
  padding: 0;
  background: #f6f8fa;
}
.card {
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 2px 12px #e6e6e6;
  padding: 32px 36px 28px 36px;
}
.graph-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 20px 0 18px 0;
}
.graph-header h2 {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
}
.subtitle {
  font-size: 1rem;
  color: #888;
  margin-left: 12px;
  font-weight: 400;
}
.controls {
  display: flex;
  gap: 12px;
}
.graph-canvas {
  width: 100%;
  height: 600px;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  background-color: #fafafa;
  position: relative;
  resize: both;
  overflow: auto;
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