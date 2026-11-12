# XR hand tracking

独立的手追踪仓库，分别生成核心与可选 Quest 后端两个 Debian 源码包。当前可选 Quest DSP 后端复用原始
DETV62、DPEV30MG backbone/fusion 权重，在 Hexagon 执行量化子图，
ARM 执行原 FP32 头与几何解码。不使用 Mercury。

`xr-device-io` / `xr-head-tracking` 保持唯一相机和 IMU 所有权。
本服务只收匹配曝光阶段的只读 memfd，输出手姿态和手势，不打开相机。
慢消费者被断开，手推理不阻塞头部生产者。服务失败不停止 SLAM。

## 软件包与启动

- `xr-hand-tracking`：MIT 服务管理、姿态/手势 IPC、状态工具。
- `xr-hand-backend-quest-dsp`：可选 ARM64 后端；包含 CC BY-NC 4.0
  的 UmeTrack 几何适配，归非自由组件选择，原始 DSP/模型二进制不随包分发。
- `libxr-hand-dev`：唯一维护的公开 ABI、C++ 客户端、CMake `XRHand::api`。
- `xr-hexagonrpc-daemon` 由 FastRPC 仓库提供，不复制维护第二套传输代码。

核心在仓库根目录构建：`dpkg-buildpackage -A -us -uc`；可选后端在
`backends/quest/` 运行 `dpkg-buildpackage -b -us -uc`。核心源码归档明确排除
Quest 后端代码，SDK 安装到架构无关的 `share/cmake/XRHand`。交叉构建使用目标发行版的库与
shlibs 元数据；可显式提供 `XR_CROSS_LIBDIR`、`XR_CROSS_SHLIBS`。
安装不启动、启用服务或切换 XR。`openkylin-xr-hands` 仅选择组件，
`vr-hands` 组合锁固定源码提交与头文件摘要。

配置 `/etc/xr/hand-tracking.json` 的私有标定、型号运行库和模型目录。
已验证的本地模型包可用 `xr-hand-import-assets --from-bundle <目录>` 导入；
导入校验原模型身份和每个资产 SHA-256，失败不会覆盖已有资产。
每次启动再次核对锁。标定不进入源码、通用型号资产或公开包。

会话配置中显式启用：

```json
{"hand_tracking":{"enabled":true,"unit":"xr-hand-tracking.service",
"camera_socket":"/run/xr-hand-tracking/camera/input.sock",
"state_socket":"/run/xr-hand-tracking/state.sock"}}
```

会话管理器先启动已有头追踪生产者，确认真实头数据后启动手消费者；
退出时先清理手 DSP PD，再清理头部采集。可选手输入配置通过
`XR_HAND_STATE_SOCKET` 选择 Monado 手设备对；默认配置继续注册物理手柄。
只注册一对左右手角色，避免两个来源争用同一角色。
Shell 设置 `XR_SHELL_INPUT_MODE=hands` 时保留射线与落点，隐藏实体手柄外观；
捏合沿现有触发动作选中/拖动，抓握沿 squeeze 动作执行 6DoF 抓取与双手缩放。
系统保持原手柄配置可切换。Shell 从标准关节绘制掌面、圆润指节与指骨表面；不嵌入厂商手部美术资产。

## 接口

`/run/xr-hand-tracking/state.sock` 为只读语义的本机 SOCK_SEQPACKET 发布流，
每包 704 字节、小端；权威声明见 `include/xr/hand/state.h`。
`xr-hand-status --samples 10` 查看原始关节、左右手、有效性和时间戳。
消费者校验 root 发布者、版本、有限数值、源有效期，并拒绝 replay 输入。
无手、断连或过期立即释放动作。重连后不复用旧握持状态。

位置为米，姿态为 xyz + xyzw，坐标为头追踪所用的 OpenVR 原始 driver world；
从 QVR 转换使用同一个 Rz(+90°) 基变换，运行时负责 recenter/origin。
手槽 0=左、1=右，不按屏幕左右重标。grip 由手掌几何确定，aim 沿食指，
两个动态 pose component 独立传给运行时，未使用头向替代手向。
捏合按拇指/食指尖距离除以掌宽，抓握按四指弯曲程度，均有 .65/.35 迟滞。
这些是本服务交互策略，不宣称复原 Quest 私有交互策略。

21 个 landmark 顺序：thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
wrist, thumb_mcp, thumb_ip, index_mcp, index_pip, index_dip, middle_mcp,
middle_pip, middle_dip, ring_mcp, ring_pip, ring_dip, pinky_mcp, pinky_pip,
pinky_dip, palm_center。这是模型输出的组件间传输格式，v1 保持兼容；并非模型或权重的所有权声明。

SDK 的 `joints.hpp` 将这些位置转换为 OpenXR 默认顺序的 26 关节。
五个掌骨位置按已解码掌部几何估计，`Joint::inferred` 明确标记；
朝向由掌背法向和相邻骨段建立，半径按掌宽估计。这五点不是新增神经检测结果。
实际运行时由 Monado OpenVR 适配提供 31 根父坐标骨骼与独立原始 grip/aim，
SteamVR 的 `XR_EXT_hand_tracking` 再提供标准 26 关节 API、半径、有效性和空间变换。
适配声明 partial skeletal tracking，避免声称所有关节具有独立传感器精度。
31 骨骼的掌腕偏移来自实测掌腕位置，不沿用实体手柄的默认手腕偏移。

Shell 启用扩展后分别创建左右手 tracker，在同一 recenter 后空间与预测显示时间查询关节。
模型与射线使用该空间，捏合/抓握沿已存在的 trigger/squeeze 6DoF 路由进入桌面，
失踪时关节模型隐藏，动作释放并取消捕获；不会用默认张手姿态冒充有效追踪。

标准参考：[Khronos 手关节接口](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrHandJointLocationEXT.html)、
[Valve 31 骨骼约定](https://github.com/ValveSoftware/openvr/wiki/Hand-Skeleton)。

保存画面可用于离线原权重解码与手势映射核对；replay 永不作为实时输入进入桌面。
生产日志轮转到有限大小，默认不保存相机像素或神经中间张量。

调试配置 `exercise_inactive: true` 可在无手时用真实当前像素执行全部子图，
输出仍标为无效；用于测量计算链路，正常配置应保持关闭。

`xr-hand-status --follow` 输出凭据校验后的只读 JSON 行供调试后端订阅；
`observer_schema=1`，时间为 CLOCK_BOOTTIME，不打开相机或启动服务。
安装及升级均不自动启停运行中的手服务；会话管理器负责切换。
RuntimeDirectory 保留 IPC 命名空间；相机/状态监听 socket 属于手消费者，
头生产者仅导出已有共享帧并重连，不因消费者退出而释放自己的相机。
