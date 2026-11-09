<script setup>
import { ref } from 'vue'
import AttendanceView from './components/AttendanceView.vue'
import PersonsView from './components/PersonsView.vue'

const activeTab = ref('attendance')
</script>

<template>
  <div class="app">
    <header class="header">
      <h1>人脸考勤系统</h1>
      <span class="subtitle">Web 管理端</span>
    </header>

    <nav class="tabs">
      <button
        :class="{ active: activeTab === 'attendance' }"
        @click="activeTab = 'attendance'"
      >考勤记录</button>
      <button
        :class="{ active: activeTab === 'persons' }"
        @click="activeTab = 'persons'"
      >人员与样本</button>
    </nav>

    <main class="content">
      <AttendanceView v-if="activeTab === 'attendance'" />
      <PersonsView v-else />
    </main>
  </div>
</template>

<style>
:root {
  --bg: #fafafa;
  --card: #ffffff;
  --border: #ececec;
  --text: #333333;
  --muted: #999999;
  --accent: #4a6cf7;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
}

.app { max-width: 1080px; margin: 0 auto; padding: 24px; }

.header { margin-bottom: 20px; }
.header h1 { font-size: 24px; font-weight: 600; }
.subtitle { color: var(--muted); font-size: 14px; margin-left: 12px; }

.tabs { display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid var(--border); }
.tabs button {
  background: none; border: none; padding: 10px 20px;
  font-size: 15px; color: var(--muted); cursor: pointer;
  border-bottom: 2px solid transparent;
}
.tabs button.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 500; }

.content { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 24px; }
</style>
