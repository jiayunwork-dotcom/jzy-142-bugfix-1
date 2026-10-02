# 弹性反应谱计算与谱匹配作业服务

把一批强震加速度记录提交上来，后台用 Newmark-β 逐步积分计算每条记录的
弹性反应谱（SD / SV / SA / PSV / PSA），再按调用方给出的设计谱做
周期区间内的对数误差最优缩放与匹配筛选，输出排序、前 N 条平均谱及
逐点比值。上传、计算、匹配都是异步作业，可查进度 / 部分结果 / 取消；
记录、作业状态与结果全部落 SQLite，进程重启后已完成结果照常可读，
执行中的作业标记为中断。

- Python 3.12 + Flask + NumPy
- 积分器、反应谱、匹配优化均为自实现（仅线性代数借助 NumPy）
- 容器：`python:3.12-slim`，gunicorn 启动，SQLite 在挂载卷

---

## 1. 目录结构

```
app/
  errors.py       # 领域异常（逐条可读错误）
  parsing.py      # 记录文本解析（两列 / 单列+步长、文件头元数据）
  baseline.py     # 基线校正（none/mean/linear）与速度位移积分
  integrator.py   # Newmark-β 积分器（向量化、自动加密、自由振动段）
  spectrum.py     # 弹性反应谱 SD/SV/SA/PSV/PSA、零周期点、PGA/PGV/PGD
  matching.py     # 设计谱对数插值、最优缩放、越限剔除、排序、平均谱比值
  jobs.py         # spectrum / match 两类作业的执行逻辑
  scheduler.py    # 单工作线程后台调度、崩溃恢复
  storage.py      # SQLite：records / jobs / job_items
  api/
    records.py    # 记录上传与查询
    jobs.py       # 作业创建 / 查询 / 取消
    helpers.py
  __init__.py     # Flask 应用工厂
wsgi.py           # gunicorn 入口
tests/            # pytest（解析/基线/积分器/谱/匹配/存储/调度/HTTP）
```

## 2. 快速开始

### 本地

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export SEISMIC_DB=/tmp/seismic.db
gunicorn -c gunicorn.conf.py wsgi:app
# 或开发模式：flask --app wsgi run
```

### 容器

```bash
docker compose up --build -d
curl -s http://localhost:8000/health
```

SQLite 文件位于卷 `seismic-data` 容器内 `/data/seismic.db`。

### 运行测试

```bash
pip install -r requirements-dev.txt
pytest -q
```

容器内（测试镜像含 pytest 与 tests/，运行镜像不含）：

```bash
docker build --target test -t seismic-spectrum:test .
docker run --rm seismic-spectrum:test
```

---

## 3. 上传记录

### multipart/form-data

字段名 `files` 可重复，每个文件一条；表单字段 `unit`（`m/s2|g`）整批
生效，`dt` 作为单列格式默认步长。

```bash
curl -s -X POST http://localhost:8000/api/records \
  -F 'files=@RSN1.txt' -F 'files=@RSN2.txt' \
  -F 'unit=m/s2'
```

### application/json

```json
{
  "unit": "g", "dt": 0.01,
  "records": [
    {"name": "RSN1.AT2", "content": "# Station: xxx\n0.00 0.001\n..."},
    {"name": "single.txt", "content_b64": "...", "dt": 0.005}
  ]
}
```

单条可用 `unit` / `dt` 覆盖整批默认值。

### 支持的文件格式

1. **时间–加速度两列**：`t a`，按数据核对等步长（相对容差 1e-4），
   声明 `dt` 时还要与时间列一致；
2. **序号–加速度两列**：第一列为 0,1,2,… 整数序号，此时必须显式给 `dt`；
3. **单列加速度**：必须给 `dt`（请求参数，或文件头 `# DT: 0.005 SEC`）。

空白/逗号/分号分隔均可。`# % * ! /` 开头的行视为文件头，原文保留到
`metadata.header`，并抽取 `station / component / channel / event /
dt / unit / name` 等常见关键字。

### 单位（必须声明）

加速度单位只接受 **`m/s2` 或 `g`**（按 9.80665 换算）。未声明单位的
记录单条标记为 `error`，不影响同批其他记录。`gal / cm/s2` 直接拒绝
（量级容易搞错），请先换算再上传。

### 逐条错误（绝不让整批一起失败）

空文件、非数值行、列数不一致、步长不一致、单位缺失或无法识别、
NaN/Inf、超过 20 万点等，都在对应记录上给出带行号的可读错误，失败
记录同样入库（`status=error`），便于事后排查。响应形如：

```json
{"uploaded": 3, "ready": 1, "failed": 2,
 "records": [
   {"name": "bad.txt", "status": "error",
    "error": "第 2 行含有非数值数据 'oops'，请检查文件内容或分隔符",
    "record_id": "rec_bad..."},
   ...
 ]}
```

限制：一批最多 **200** 条，单条最多 **200 000** 点（超出整请求 400
或该条 error）。

记录 ID 由文件名 + 加速度字节 + 步长/单位的 SHA-256 前 16 位决定，
重复上传同一文件得到同一 ID。

---

## 4. 创建作业

### 4.1 反应谱作业

`POST /api/jobs/spectrum`

```json
{
  "record_ids": ["rec_...", "rec_..."],
  "periods": [0.0, 0.02, 0.05, 0.1, 0.5, 1.0, 3.0, 6.0],
  "dampings": [0.02, 0.05],
  "method": "average_acceleration",
  "instability_policy": "refine",
  "baseline_mode": "mean"
}
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `periods` | `logspace(log10 0.02, log10 6, 100)` | 严格递增、非负；允许包含 0 |
| `dampings` | `0.05` | 数值或列表，均须 ∈ [0,1] |
| `method` | `average_acceleration` | `average_acceleration`（γ=1/2,β=1/4，无条件稳定）或 `linear_acceleration`（γ=1/2,β=1/6） |
| `instability_policy` | `refine` | `refine`（线性插值自动加密）/ `reject`（失稳即该条报错） |
| `baseline_mode` | `mean` | `none` / `mean` / `linear`，见 §7 |

### 4.2 匹配作业

`POST /api/jobs/match`，在谱作业参数之外增加：

```json
{
  "record_ids": ["rec_a", "rec_b"],
  "periods": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0],
  "dampings": [0.05],
  "target_damping": 0.05,
  "design_spectrum": {
    "periods": [0.01, 0.1, 0.5, 1.0, 6.0],
    "values":  [0.20, 0.60, 0.60, 0.35, 0.10],
    "unit": "g"
  },
  "t1": 0.1, "t2": 2.0,
  "s_min": 0.5, "s_max": 3.0,
  "top_n": 7,
  "quantity": "psa",
  "bounds_policy": "reject"
}
```

- 设计谱控制点周期严格递增、周期与谱值严格为正（对数插值要求）；
- 匹配区间 `[t1,t2]` 必须落在设计谱定义范围内；
- `quantity`：`psa`（默认，伪加速度）或 `sa`；
- 设计谱单位为 `g` 时记录谱自动换算到 g 后再匹配；
- `bounds_policy`：`reject`（越限即从候选剔除并注明原因）或 `clamp`
  （取边界系数重算误差后仍参与排序）。

### 查询与取消

| 接口 | 说明 |
|---|---|
| `GET /api/jobs` | 最近作业（不含大结果体） |
| `GET /api/jobs/<id>` | 状态、进度、完成后的完整结果 |
| `GET /api/jobs/<id>/items` | 逐条作业项，执行中即可看到已完成部分 |
| `POST /api/jobs/<id>/cancel` | 取消排队 / 执行中的作业 |

作业状态：`queued → running → completed / failed / cancelled`；
进程重启时上次 `running` 的作业变为 `interrupted`（结果不可用，
请重新提交），`queued` 作业继续排队执行。

取消响应（字段只新增、路径不变）：

- **排队中（queued）**：作业没有执行线程，取消在一个事务里直接终态，
  返回 `200 {"status":"cancelled","cancel_requested":true}`；
- **执行中（running）**：返回 `200 {"status":"running","cancel_requested":
  true}`。响应只表示**取消请求已确认落库**，作业暂时仍是 running，
  执行线程在下一个检查点收尾后变为 cancelled；调用方轮询 `GET` 即可。
  关键承诺：一旦拿到这个 200，作业最终**只能**是 cancelled，任何收尾
  都不能把它改写成 completed；
- **已在终态**：如实回 `409`，错误体带 `status` 字段说明当前终态
  （completed / failed / cancelled / interrupted）。特别是作业已经
  算完并写完结果后到达的取消，返回 409 而不是先成功后翻盘。

作业详情与列表新增只读字段 `cancel_requested`（布尔，取消请求是否已
登记）。

---

## 5. 输出内容

每条谱结果包含（每个阻尼比一组）：

- `periods`；
- `sd`（相对位移峰值，m）、`sv`（相对速度峰值，m/s）、
  `sa`（绝对加速度峰值，m/s²）；
- `psv = ω·SD`、`psa = ω²·SD`；
- `sa_g`、`psa_g`：以 g 计，便于直接对规范谱；
- `method`、`dt_used`、`refined`、`refine_factor`、
  `unstable_at_input_dt`（逐周期布尔掩码）、`tail_steps`——**明确写出
  线性加速度法是否触发了加密、加密倍数与实际积分步长**；
- 记录层面：`pga / pgv / pgd`（及 `pga_g`）、采用的 `baseline` 方式、
  自由振动段信息 `free_vibration.included=true, periods_factor=1.25`。

匹配结果：`ranking`（rank、record_id、scale、无约束最优
`scale_optimal`、`mse`、`rmse_log`、是否越限）、`excluded`（越限记录
及原因）、`average_spectrum`（前 N 条缩放后谱的算术平均与几何平均）、
`ratios`（设计谱控制点上的平均谱/几何平均谱与设计谱的**逐点比值**）。

---

## 6. 数值方法

### 6.1 运动方程与 Newmark-β

单位质量单自由度线弹性体系：

```
ẍ + 2ξω ẋ + ω²x = -a_g(t)
```

静止起步 `x(0)=v(0)=0`，故 `a(0) = -a_g(0)`。每个时间步解等效
静力方程 `K̂ x_{n+1} = RHS`，再由 Newmark 假设回代 `a_{n+1}`、
`v_{n+1}`；所有振荡器（不同周期/同一阻尼）作为数组最后一维做向量化
递推。绝对加速度取 `a_rel + a_g`。

### 6.2 稳定性与自动加密

- 平均加速度法（β=1/4）无条件稳定；
- 线性加速度法（β=1/6）条件稳定，无阻尼临界
  `ωΔt ≤ √12`，即 **`Δt/T ≤ √12/(2π) ≈ 0.55133`**（这是最严的
  无阻尼判据，对 ξ≥0 同样安全）；
- 越限时 `refine` 策略按 `ceil((Δt/Tmin)/0.55133)` 把输入**线性插值**
  加密为整数倍子步——线性加速度法本身假设区间内加速度线性变化，
  因此该加密与其离散假设严格一致，不引入额外近似；`reject` 策略则
  直接让该条记录失败并在错误中写明周期、Δt/T 与上限。

### 6.3 自由振动段

记录结束并不意味着响应峰值已经出现。统一在强迫段后补算
**1.25 个（本批最长）自振周期**的零输入自由振动（至少 1 步）并
纳入 SD/SV/SA 峰值统计。

为什么 1.25 个周期对 0≤ξ≤1 都足够：自由振动可写成
`x = e^{-ξωt}(A sin ωd t + B cos ωd t)`，相邻局部极值幅值以
`e^{-ξωΔt}` 衰减，而首个极值落在 `t∈[0, Td/2]` 内。自由振动
**初始包络**为

```
R0 = sqrt(x0² + ((v0+ξωx0)/ωd)²),  ωd = ω sqrt(1-ξ²)
```

它就是自由段 `|x|` 的理论最大值（ξ=0 时恰为一个自振周期内扫到的
振幅），故补满略大于一个周期（取 1.25 留边界余量）即可保证不遗漏
峰值；有阻尼时峰值出现更早、后续指数衰减，更长尾部不会改变峰值。
这也保证「在记录末尾再补一段零，谱值不变」（见测试）。

### 6.4 零周期点与极限行为

- 周期列表允许含 0：该点 `SA = PSA = PGA`，`SD = SV = PSV = 0`；
- T→0 时 SA→PGA（体系刚度无穷大，质量与地面同加速度）；
- T→∞ 时 SD→PGD（体系近不动，相对位移即地面位移），要求记录先做
  基线处理（见 §7）。

### 6.5 谱匹配优化

统计网格取记录谱周期点 ∩ [t1,t2]，并补入区间端点；设计谱在该网格
按**对数–对数插值**。对缩放系数 s 最小化

```
Σ_j [ ln(s A_j) - ln D_j ]²
```

令导数为零得闭式最优解（各点比值的几何平均）：

```
s* = exp( mean_j ln(D_j / A_j) )
mse = mean_j ( ln(s A_j / D_j) )²
```

这是一维无约束凸问题的精确解（无需迭代优化器）；`clamp` 时取
`s = clip(s*, s_min, s_max)` 后重算 mse。记录整体乘 k 时，
`A→kA ⇒ s*→s*/k`，mse 不变（测试逐条验证）。

### 6.6 确定性

相同输入 → 相同结果：记录 ID 内容寻址；匹配排序在 mse 相同时按
record_id 二次排序；结果 JSON 键排序序列化；不依赖当前时间作为
随机源。重复提交同一批作业，结果除作业 ID/时间戳外完全一致。

---

## 7. 基线处理（重点说明）

强震加速度记录常带微小零点偏移/慢漂移。两次梯形积分求地面位移时，
恒定偏移 ε 会产生 `½ ε t²` 量级的**虚假位移漂移**，直接破坏
「长周期 SD 趋于 PGD」。因此每条记录计算前统一做基线校正，方式写入
请求参数 `baseline_mode` 与结果 `baseline` 字段：

| 方式 | 操作 | 用途 |
|---|---|---|
| `none` | 不处理 | 记录已离线校正过时使用 |
| `mean`（默认） | 减去全程加速度均值（去直流） | 一般记录；处理后末端地面速度回到 ≈0 |
| `linear` | 最小二乘拟合一次趋势线并扣除（同时去直流 + 去线性漂移） | 末端速度/位移明显不回零的记录 |

地面速度、位移均由复合梯形积分、零初始条件得到：

```
v[0]=d[0]=0,  v[i]=v[i-1]+(a[i-1]+a[i])·Δt/2  （再积分一次得 d）
PGV = max|v|,  PGD = max|d|
```

基线校正属于线性运算：在默认 `mean` 下记录整体乘 k 时谱仍线性
（仅有求和顺序的浮点舍入，测试以 1e-10 容差验证；`none` 模式
逐位线性）。

---

## 8. 作业调度、取消与持久化

### 8.1 基本结构

- 单工作线程**串行**执行作业（SQLite 单写；NumPy/BLAS 负责计算并行）；
- 单条记录失败只标记该作业项 `error/skipped`（带可读原因），作业整体
  照常 `completed`，`result.summary` 给出 done/error/skipped 计数；
- SQLite（WAL）三张表：`records`（含加速度 BLOB、元数据 JSON、
  status/error）、`jobs`（参数/进度/结果 JSON/时间戳/取消标志）、
  `job_items`（逐条状态与结果，执行中即可查部分结果）；
- 进程重启：`running → interrupted`（注明原因），`queued` 继续执行；
  `completed` 结果长期可查。

gunicorn 刻意配置为 **1 worker + 多线程**，避免多进程各自启动调度
线程重复领取队列（见 `gunicorn.conf.py` 注释）。

### 8.2 终态仲裁：取消请求与执行线程谁说了算

**决策：终态只由执行线程在唯一入口 `finalize_job` 的一条 SQLite 事务
里落定；取消接口只登记请求，绝不直接把 running 作业写成终态。**

`jobs.cancel_requested` 是与作业状态分离的请求标志（新增列，老库
启动时自动 `ALTER TABLE` 加列）。取消 running 作业只把该标志置 1；
执行线程在各检查点轮询标志，收尾时在同一事务内做二选一仲裁：

- 事务开始时标志已置 → `cancelled`：丢弃作业结果，把仍处于
  pending/running 的作业项全部置为 `cancelled`；
- 否则 → `completed`：正常写结果。

SQLite 的库级写锁把两个事务严格串行化：取消标志的提交与结果的提交
谁先进入数据库，仲裁就判给谁。因此「取消确认后翻盘为 completed」在
机制上不可能发生；反过来，完成先提交后取消接口只能拿到 409。

**理由**：跨线程的「最终事实」必须只有一个写入点，否则任何先给
running 作业改状态、再指望工作线程「看到后别动」的方案，都留下了
改状态与收尾之间的翻盘窗口（旧实现正是如此：取消接口直接写
`status='cancelled'`，工作线程轮询粒度又只在条目之间，于是正在算的
最后一条算完后照常 `complete_job`，把状态覆盖回 completed）。

**代价**：取消 running 作业的 HTTP 响应不能再直接写 `status=
cancelled`（那一刻作业确实还在跑），改为返回 `status="running",
"cancel_requested": true`，调用方需要轮询到真正终态。字段名与路径
不变，只新增 `cancel_requested`；只认旧响应里 `status` 字段的老
调用方在取消 running 作业时会看到 `running`，语义即「请求已收到」。

排队作业没有执行线程，不存在竞争：取消在同一事务里直接判 cancelled
并把全部作业项置为 cancelled，响应仍是终态。

### 8.3 取消到达时正在算的那条记录：尽快中止、结果丢弃

**决策：协作式中止，丢弃部分结果，该作业项标记为 `cancelled`。**

积分主循环每 `CANCEL_CHECK_EVERY = 256` 个时间步查询一次取消回调
（一次主键索引查询，开销可忽略），发现取消即抛内部异常 `JobCancelled`
中止本条：已算的部分峰值不落库、不返回。条目之间、匹配阶段前后另有
检查点。

**理由**：反应谱结果是一个不可分割的整体（各周期峰值 + PGA/PGV/PGD
+ 自由振动段），写出「算到 70%」的半份结果比没有结果更危险——下游
选波无法区分它和完整结果。

**代价**：响应取消后作业不会在零时间停下，最坏要等当前条目走完
≤256 个时间步（20 万点 × dt=0.005s 的作业也是毫秒级），随后才进入
cancelled 终态；已经消耗的该条 CPU 时间作废。作为对照，另一种方案
「让当前条目算完再停」会让取消延迟一整条记录的耗时（5 秒量级），
不可接受，故不采用。

### 8.4 最后一条算完、结果尚未写入的瞬间来了取消

这是唯一无法在语义上两全的边界，**判给先在数据库里落定的一方**，
只有两种合法结局，测试逐一对打验证：

1. 取消标志的事务先提交：执行线程 `finalize_job` 仲裁为 cancelled，
   完整结果丢弃，作业项仍是 done（条目结果已在其自己的事务中落库，
   可读），但作业整体无 result；
2. 完成的事务先提交：作业为 completed，取消请求随后到达，接口如实
   返回 409。

不会出现「两边都成功」或「作业消失」。选择数据库提交顺序而非任何
一方优先，是因为两个提交由同一把锁串行，天然给出确定且可复现的
裁决，不需要分布式锁或额外协调。

匹配作业在最后的排序匹配阶段（`match_batch`，毫秒级纯 Python）
前后均有检查点：阶段内到达的取消按同一规则处理——若取消先落库，
匹配结果整体丢弃（job 无 result），但逐条谱结果保留在作业项上。

### 8.5 作业项状态：新增 cancelled / interrupted

| 状态 | 含义 | 结果 |
|---|---|---|
| `pending` | 尚未开始（仅执行中可见，任何终态后不允许残留） | 无 |
| `running` | 正在计算（同上） | 无 |
| `done` | 正常完成 | 有 |
| `error` | 本条计算失败（记录问题、reject 失稳等） | 无，error 列有原因 |
| `skipped` | 记录不可用等原因跳过 | 无，error 列有原因 |
| **`cancelled`**（新增） | 因作业取消而没跑成：取消时还在 pending 的（没跑），或算到一半被中止的（结果丢弃） | 无，error 列说明是「未执行」还是「执行中中止」 |
| **`interrupted`**（新增） | 因进程在作业执行中死亡而没跑完：重启时 running 的条目（算到一半进程死了）与 pending 的条目（没轮到）都置此状态 | 无，error 列说明中断点 |

不变量（测试逐条核对）：作业一旦进入任何终态，作业项中**不再有
pending/running**；`GET /items` 的 `counts` 各状态计数之和恒等于
作业 total。`counts` 同时新增 `cancelled`、`interrupted` 两个键
（旧键不动）。

取消/中断**之前**已经 done 的作业项，其结果不回收：取消作业保留
（可在 items 接口读到，job 级 result 仍为空）；重启后同样可读。

**老调用方看到什么**：

- 只从 job 的 `result` 取结果的调用方不受影响（cancelled/interrupted
  作业本来就没有 job 级 result）；
- 从 items 里按 `status == "done"` 收集成功条目的调用方天然忽略两个
  新状态；
- 若有调用方对 items 状态做穷举分支，遇到新状态应按「终态、无结果、
  error 列有人类可读原因」保守处理；`counts` 中新增的键建议遍历时
  求和而不是按固定键名取值。

### 8.6 重启恢复的细化规则

`recover_interrupted` 在新进程启动时对上次 `running` 的作业：

- 未登记取消：作业 → `interrupted`；running 条目 → `interrupted`
  （执行中死亡），pending 条目 → `interrupted`（未执行），done 条目
  结果保留；
- **已登记取消**（取消请求已确认 200，但收尾没落库进程就死了）：
  作业仍判 `cancelled`——取消承诺不因进程死亡而失效；running 条目
  → `cancelled`（执行中中止），pending 条目 → `cancelled`（未执行）。

`queued` 作业恢复后继续排队执行，行为不变。

---

## 9. 测试覆盖（pytest，共 130+ 用例）

需求中列出的关系均有**独立测试**逐条确认：

- 记录整体乘 k：所有谱值乘 k、匹配缩放系数除以 k
  （`test_scaling_linearity*`、`test_scaled_record_scales_inverse`）；
- 纯正弦地面运动稳态放大系数与解析解
  `R(r)=1/sqrt((1-r²)²+(2ξr)²)` 在 r=0.7/1.0/1.3 相对误差 <1%
  （`test_sine_steady_state_amplification_within_1pct`）。测试信号
  用 20 周期半余弦包络平滑起步以消除**起振瞬态拍振**，并按自由振动
  初始包络 R0 的二次型最小特征值选择截断相位，保证记录突停后的自由
  振动不抬高峰值——这两个构造的物理原因在测试 docstring 中写明；
- ξ=0 时 SA 恰等于 PSA（容差 1e-10），小阻尼中短周期段两者统计接近；
- T→0 时 SA→PGA（带限信号严格验证；并注释了宽带噪声短周期 SA 高于
  PGA 是真实高频反力现象）；T→∞ 时 SD→PGD（位移脉冲构造）；
- 两种 Newmark 参数在步长足够细时收敛到同一结果（<2e-3）；
- 记录末尾补一段零，谱值不变；
- 同一批作业重复提交，结果完全一致；
- 线性加速度法稳定判据、自动加密倍数、`reject` 报错信息、
  `unstable_at_input_dt` 掩码；
- 解析：空文件、非数值行（带行号）、步长不一致、单位缺失/不支持、
  阻尼越界、周期非递增、设计谱非法、缩放越限剔除、批量上限；
- 存储往返、崩溃恢复（running→interrupted、queued 续跑）、
  完成结果跨「重启」可读、HTTP 端到端上传/建作业/进度/取消/部分结果；
- **取消与重启回归**（`test_cancel_restart.py`，可直接在容器里
  `pytest` 执行），取消落在不同时机反复多次，每次都核对作业状态、
  逐条状态、计数三者一致：
  - 单条 20 万点记录取消 ×12：终态只能是 cancelled，等待远超计算
    耗时后不翻盘，作业项 cancelled、无结果；
  - 三条记录取消落在第一条 / 最后一条 ×8：没跑的条目 cancelled，
    已 done 的结果保留可读，`progress == done+error+skipped`；
  - 「最后一条算完、结果未提交」边界：取消先到 / 完成先到各 ×10 的
    确定性交错 + ×40 双线程自由竞争，裁决只可能是「取消成功且最终
    cancelled」或「完成在先、取消 409」，二者必居其一且自洽；
  - 匹配作业取消落在逐条算谱阶段 ×6、最后的排序匹配阶段 ×10
    （经 `_before_match_hook` 在 match 开始瞬间登记取消）；
  - 执行中模拟进程死亡（第一条 / 最后一条在跑）×6：重启后
    interrupted，running/pending 全部离开非终态并标明中断点，
    done 结果重启后可读；
  - 取消已确认后进程死亡：重启仍判 cancelled；同库 queued 作业续跑
    完成；
  - 积分器取消回调在时间步检查点快速抛 `JobCancelled` 的单元测试。

  时序不依赖「sleep 多久」：取消与「崩溃」都挂在工作线程的同步提交
  点（`set_item_running` 之后同一拍）触发，保证取消必然落在指定条目
  的计算中，结果可稳定重复。

## 10. 容量与性能

- 批量 ≤ 200 条、单条 ≤ 200 000 点（上传与建作业两处都校验）；
- 典型强震记录（数千点、100 周期点、5% 阻尼）单条亚秒级；
- 20 万点 × 100 周期的极端组合每步是 100 维向量运算，主要成本为
  Python 时间步循环，属后台批处理可接受范围；积分热点全部在 NumPy。
