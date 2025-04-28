如果你想将 `libportaudio2` 及其所有依赖包下载到本地，然后在没有网络连接的情况下进行离线安装，下面是详细的步骤。

### 1. 下载所需的 `.deb` 包（在一台有网络的机器上）
首先，你需要在一台有网络连接的机器上下载所需的 `.deb` 包。可以通过 `apt-get` 或 `apt` 下载这些包并其依赖项。

#### 步骤：

1. **更新包列表**：
   
   在有网络的机器上，首先更新包列表：

   ```bash
   sudo apt update
   ```

2. **下载包及其所有依赖项**：
   
   使用 `apt-get download` 命令来下载指定包及其所有依赖项。这里是以 `libportaudio2` 为例：

   ```bash
   sudo apt-get install --download-only libportaudio2
   ```

   该命令不会安装任何东西，只会将所需的 `.deb` 包下载到本地。

   默认情况下，`.deb` 包会被下载到 `/var/cache/apt/archives/` 目录。

   你可以进入该目录并查看下载的包：

   ```bash
   cd /var/cache/apt/archives/
   ls libportaudio2*.deb
   ```

3. **下载所有依赖项**：
   
   你可以使用 `apt-get` 获取所有依赖项的 `.deb` 包。如果你想一次性下载所有依赖包，可以使用以下命令：

   ```bash
   sudo apt-get install --download-only libportaudio2
   ```

   这样，`libportaudio2` 及其所有依赖包都会被下载到 `/var/cache/apt/archives/` 目录。

4. **复制 `.deb` 包**：
   
   将下载的 `.deb` 包复制到一个 USB 驱动器或其他外部存储设备上，以便在没有网络的目标机器上安装。

### 2. 离线安装包及依赖（在没有网络的机器上）
一旦你将 `.deb` 包传输到目标机器（没有网络连接的 Debian 11 系统），你可以按照以下步骤进行离线安装。

#### 步骤：

1. **复制 `.deb` 包**：
   
   将下载的 `.deb` 包从 USB 驱动器或其他存储设备复制到目标机器上。例如，你可以将它们放到 `/home/user/debs/` 目录。

2. **安装 `.deb` 包**：

   进入存放 `.deb` 包的目录，然后使用 `dpkg` 命令安装它们：

   ```bash
   sudo dpkg -i /home/user/debs/*.deb
   ```

   这将安装目录中的所有 `.deb` 包。

3. **修复缺失的依赖关系**：
   
   在安装过程中，`dpkg` 可能会报告某些依赖关系缺失。在这种情况下，你可以运行以下命令修复缺失的依赖项：

   ```bash
   sudo apt --fix-broken install
   ```

   这将尝试修复所有丢失的依赖项。

4. **验证安装**：
   
   安装完成后，检查 `libportaudio2` 是否已正确安装：

   ```bash
   dpkg -l | grep libportaudio2
   ```

   你也可以验证其他依赖是否成功安装。

### 3. 备选方法：使用 `apt-offline` 工具
`apt-offline` 是一个工具，它可以帮助你在没有网络的机器上下载和安装软件包及其依赖。你可以通过以下步骤使用它：

1. **在有网络的机器上下载所需包**：

   安装 `apt-offline`：

   ```bash
   sudo apt-get install apt-offline
   ```

   使用以下命令创建一个签名文件，它会列出要安装的包及其所有依赖：

   ```bash
   apt-offline set offline.sig --update
   apt-offline get offline.sig --bundle offline.zip
   ```

2. **在离线机器上安装**：

   将 `offline.zip` 文件传输到离线机器，然后使用 `apt-offline` 安装所有包：

   ```bash
   sudo apt-offline install offline.zip
   ```

这样，你就可以在没有网络的机器上完成所有依赖包的安装了。

### 总结
- 使用 `apt-get` 和 `--download-only` 来下载 `libportaudio2` 及其依赖。
- 将 `.deb` 包复制到离线机器后，使用 `dpkg -i` 安装它们。
- 如果遇到依赖问题，使用 `apt --fix-broken install` 来修复。

这样就能在没有网络连接的 Debian 10 系统上成功离线安装 `libportaudio2` 及其依赖。