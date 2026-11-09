<script setup>
import { ref, onMounted, watch } from 'vue'
import * as echarts from 'echarts'

const API = 'http://127.0.0.1:8000'

const records = ref([])
const stats = ref(null)
const filterDate = ref('')
const chartEl = ref(null)
let chart = null

async function loadRecords() {
  const url = filterDate.value
    ? `${API}/api/attendance?date=${filterDate.value}`
    : `${API}/api/attendance`
  const res = await fetch(url)
  const json = await res.json()
  records.value = json.data
}

async function loadStats() {
  const res = await fetch(`${API}/api/stats`)
  stats.value = await res.json()
  renderChart()
}

function renderChart() {
  if (!stats.value || !chartEl.value) return
  if (!chart) chart = echarts.init(chartEl.value)
  const dates = stats.value.by_date.map(d => d.d)
  const counts = stats.value.by_date.map(d => d.c)
  chart.setOption({
    tooltip: { trigger: 'axis' },
    grid: { left: 40, right: 20, top: 30, bottom: 30 },
    xAxis: { type: 'category', data: dates },
    yAxis: { type: 'value', minInterval: 1 },
    series: [{
      type: 'bar',
      data: counts,
      itemStyle: { color: '#4a6cf7', borderRadius: [4, 4, 0, 0] },
      barWidth: '40%'
    }]
  })
}

function exportExcel() {
  const url = filterDate.value
    ? `${API}/api/attendance/export?date=${filterDate.value}`
    : `${API}/api/attendance/export`
  window.open(url, '_blank')
}

function fmtSim(v) { return v == null ? '-' : v.toFixed(3) }

onMounted(() => {
  loadRecords()
  loadStats()
})
watch(filterDate, loadRecords)
</script>

<template>
  <div>
    <!-- 统计卡片 -->
    <div class="stats-row" v-if="stats">
      <div class="stat-card">
        <div class="num">{{ stats.total_persons }}</div>
        <div class="label">已录入人员</div>
      </div>
      <div class="stat-card">
        <div class="num">{{ stats.total_attendance }}</div>
        <div class="label">考勤记录数</div>
      </div>
    </div>

    <!-- 图表 -->
    <div ref="chartEl" class="chart" v-if="stats && stats.by_date.length"></div>

    <!-- 工具栏 -->
    <div class="toolbar">
      <input
        v-model="filterDate"
        type="date"
        placeholder="按日期筛选"
        class="date-input"
      />
      <button class="btn" @click="exportExcel">导出 Excel</button>
    </div>

    <!-- 考勤表格 -->
    <table class="table">
      <thead>
        <tr>
          <th>学号</th>
          <th>姓名</th>
          <th>时间</th>
          <th>时段</th>
          <th>相似度</th>
          <th>清晰度</th>
          <th>亮度</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="r in records" :key="r.id">
          <td>{{ r.student_no }}</td>
          <td>{{ r.name }}</td>
          <td>{{ r.ts }}</td>
          <td>{{ r.session_key.split('-').pop() }}</td>
          <td>{{ fmtSim(r.similarity) }}</td>
          <td>{{ r.sharpness?.toFixed(0) ?? '-' }}</td>
          <td>{{ r.brightness?.toFixed(0) ?? '-' }}</td>
        </tr>
        <tr v-if="!records.length">
          <td colspan="7" class="empty">暂无考勤记录</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.stats-row { display: flex; gap: 16px; margin-bottom: 20px; }
.stat-card {
  flex: 1; padding: 20px; border: 1px solid #ececec; border-radius: 8px;
  text-align: center;
}
.num { font-size: 32px; font-weight: 600; color: #4a6cf7; }
.label { color: #999; font-size: 13px; margin-top: 4px; }

.chart { height: 260px; margin-bottom: 20px; }

.toolbar { display: flex; gap: 12px; margin-bottom: 16px; align-items: center; }
.date-input {
  padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px;
}
.btn {
  padding: 8px 16px; background: #4a6cf7; color: #fff; border: none;
  border-radius: 6px; cursor: pointer; font-size: 14px;
}
.btn:hover { background: #3a5ce0; }

.table { width: 100%; border-collapse: collapse; }
.table th, .table td {
  padding: 10px 12px; text-align: left; border-bottom: 1px solid #f0f0f0;
  font-size: 14px;
}
.table th { color: #999; font-weight: 500; background: #fafafa; }
.empty { text-align: center; color: #ccc; padding: 40px; }
</style>
