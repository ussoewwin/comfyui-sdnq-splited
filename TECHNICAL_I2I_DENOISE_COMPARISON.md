# 標準KSampler vs SDNQSamplerV2: i2iとDenoise実装の完全技術解説

## 目次
1. [概要とアーキテクチャの根本的違い](#概要とアーキテクチャの根本的違い)
2. [Denoise強度の制御メカニズム](#denoise強度の制御メカニズム)
3. [Image-to-Image (i2i) 処理の実装](#image-to-image-i2i-処理の実装)
4. [Flux2特有の特殊実装](#flux2特有の特殊実装)
5. [非Fluxパイプラインの処理](#非fluxパイプラインの処理)
6. [技術的課題と解決策](#技術的課題と解決策)
7. [実装の整合性と設計判断](#実装の整合性と設計判断)

---

## 1. 概要とアーキテクチャの根本的違い

### 1.1 標準KSamplerのアーキテクチャ

標準のComfyUI KSamplerは、**ComfyUIの内部モデル表現**（`ModelPatcher`、`CLIP`、`VAE`）を直接操作します。

**処理フロー:**
```
入力LATENT → VAEデコード → 画像テンソル → ノイズ追加 → VAEエンコード → 拡散処理 → VAEデコード → 出力IMAGE
```

**特徴:**
- ComfyUIの内部API（`sampling_function`、`model_function`）を使用
- 伝統的な拡散モデル（SDXL、SD1.5）向けに最適化
- `denoise`パラメータは**ステップ数の削減**として機能
  - `effective_steps = steps * denoise`
  - 例: `steps=20, denoise=0.5` → 10ステップで処理

### 1.2 SDNQSamplerV2のアーキテクチャ

SDNQSamplerV2は、**diffusersライブラリのパイプライン**を直接使用します。

**処理フロー:**
```
入力LATENT → パイプライン判定 → Flux2/非Flux分岐 → パイプライン固有処理 → diffusers.__call__() → 出力IMAGE
```

**特徴:**
- diffusersの`DiffusionPipeline`を直接呼び出し
- Flow Matchingモデル（Flux2）と伝統的拡散モデルの両方に対応
- `denoise`パラメータの解釈がパイプラインタイプによって**根本的に異なる**

---

## 2. Denoise強度の制御メカニズム

### 2.1 標準KSamplerのDenoise実装

**実装ロジック:**
```python
# 標準KSampler（概念的な実装）
effective_steps = int(steps * denoise)
# denoise=0.5, steps=20 → effective_steps=10

# 処理は最初の10ステップのみ実行
# 残りの10ステップはスキップされる
```

**動作原理:**
- `denoise`は**処理ステップ数の比率**を制御
- `denoise=1.0`: 全ステップ実行（完全なノイズ除去）
- `denoise=0.5`: 半分のステップのみ実行（元画像に近い）
- ステップ数の削減により、**ノイズ除去が不完全**になり、元画像の特徴が残る

**制約:**
- ステップ数が少ないと品質が低下する可能性
- 低い`denoise`値では、ノイズ除去が不十分になる

### 2.2 SDNQSamplerV2のDenoise実装（非Fluxパイプライン）

**実装ロジック:**
```python
# SDNQSamplerV2 - 非Fluxパイプライン（nodes/samplerv2.py:573-574）
if (not is_flux_family) and strength is not None and ("strength" in call_params):
    pipeline_kwargs["strength"] = strength  # strength = denoise
```

**動作原理:**
- diffusersの標準`strength`パラメータを使用
- `strength`は**ノイズレベルの初期値**を制御
  - `strength=1.0`: 完全なノイズから開始（text-to-image相当）
  - `strength=0.5`: 中間的なノイズレベルから開始
  - `strength=0.2`: 元画像に近い状態から開始
- **ステップ数は変更されず**、初期ノイズレベルが調整される

**標準KSamplerとの違い:**
- ステップ数は常に`steps`のまま（削減されない）
- 初期ノイズレベルの調整により、より**滑らかなdenoise制御**が可能
- 品質の低下が少ない

### 2.3 SDNQSamplerV2のDenoise実装（Flux2パイプライン）

**実装ロジック:**
```python
# SDNQSamplerV2 - Flux2パイプライン（nodes/samplerv2.py:502-519）
if is_flux_family and pipeline_type in ["Flux2Pipeline", "FluxPipeline"] and strength is not None:
    # sigma_end = 0.0 に固定（重要！）
    sigma_end = 0.0
    sigma_start = float(strength)  # denoise値をそのまま使用
    
    # sigma配列を生成
    sigmas = np.linspace(sigma_start, sigma_end, req_steps, dtype=np.float32).tolist()
    pipeline_kwargs["sigmas"] = sigmas
```

**Flow Matchingの基本原理:**
Flow Matchingでは、**sigma（ノイズレベル）が直接混合比を制御**します：

```
x_t = sigma * noise + (1 - sigma) * x0
```

- `sigma=1.0`: 完全なノイズ（`x_t = noise`）
- `sigma=0.5`: ノイズと元画像の50:50混合
- `sigma=0.0`: 元画像そのもの（`x_t = x0`）

**sigma_end = 0.0 の重要性:**
```python
# 510-511行目のコメントより
# Use 0.0 terminal sigma so low denoise can truly stay close to the init image.
# (Using 1/steps creates a "noise floor" that can make denoise feel inverted at low step counts.)
sigma_end = 0.0
```

**なぜ0.0が重要か:**
- `sigma_end`が`1/steps`（例: 0.05）の場合、最終的に5%のノイズが残る
- これにより、低い`denoise`値でも完全に元画像に近づけない
- `sigma_end=0.0`により、**完全に元画像に収束可能**

**標準KSamplerとの根本的違い:**
1. **ステップ数は変更されない**: `req_steps = int(steps)`（384行目）
2. **sigmaスケジュールで制御**: `denoise`値が直接`sigma_start`になる
3. **数学的に正確な混合**: Flow Matchingの式に基づく

---

## 3. Image-to-Image (i2i) 処理の実装

### 3.1 標準KSamplerのi2i処理

**処理フロー:**
```
入力IMAGE → VAEエンコード → LATENT取得
→ ノイズ追加（denoiseに応じて） → 拡散処理 → VAEデコード → 出力IMAGE
```

**実装の特徴:**
- ComfyUIの内部VAEを使用
- 入力画像をlatent空間にエンコード
- ノイズを追加して拡散処理を開始
- `denoise`値に応じてノイズ量を調整

**制約:**
- ComfyUIのモデル構造に依存
- パイプラインの種類による差異を考慮しない

### 3.2 SDNQSamplerV2のi2i処理（非Fluxパイプライン）

**実装ロジック:**
```python
# nodes/samplerv2.py:575-593
elif (not is_flux_family) and isinstance(latent_image, dict) and latent_image.get("vae") is not None:
    # 入力latentをそのまま使用
    init_latents = samples
    # dtypeとdeviceを調整
    pipeline_kwargs["latents"] = init_latents
    pipeline_kwargs["strength"] = strength  # denoise値
```

**動作原理:**
- 入力latentをそのまま`pipeline_kwargs["latents"]`に渡す
- `strength`パラメータでノイズレベルを制御
- diffusersパイプラインが内部でノイズを追加

**標準KSamplerとの違い:**
- diffusersの標準APIを使用
- パイプラインが内部でノイズ追加を処理

### 3.3 SDNQSamplerV2のi2i処理（Flux2パイプライン） - 核心実装

**実装ロジック（完全版）:**
```python
# nodes/samplerv2.py:502-564

# ステップ1: sigmaスケジュールの生成
sigma_end = 0.0
sigma_start = float(strength)  # denoise値
sigmas = np.linspace(sigma_start, sigma_end, req_steps, dtype=np.float32).tolist()
pipeline_kwargs["sigmas"] = sigmas

# ステップ2: 画像の前処理
image_tensor = pipeline.image_processor.preprocess(
    pil_cond, height=img_h, width=img_w, resize_mode="crop"
)
image_tensor = image_tensor.to(device=generator_device, dtype=pipeline.vae.dtype)

# ステップ3: VAEエンコード（x0の取得）
x0 = pipeline._encode_vae_image(image=image_tensor, generator=generator)
# x0 shape: [B, 128, H', W'] (unpacked latent)

# ステップ4: timestepsの計算
from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_timesteps, compute_empirical_mu
image_seq_len = int(token_h * token_w)
mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=req_steps)
timesteps, _ = retrieve_timesteps(
    pipeline.scheduler, req_steps, generator_device, sigmas=sigmas, mu=mu
)

# ステップ5: ノイズの追加とスケーリング
t0 = timesteps[0].expand(x0.shape[0]).to(device=x0.device)
noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
x_t = pipeline.scheduler.scale_noise(sample=x0, timestep=t0, noise=noise)
# x_t = sigma_start * noise + (1 - sigma_start) * x0 の実装

# ステップ6: 初期latentの設定
pipeline_kwargs["latents"] = x_t
```

**なぜこの実装が必要か:**

1. **Flux2の`image`パラメータの特殊性:**
   - Flux2では`image`は**追加の条件付けトークン**として機能
   - これだけでは**真のi2iにならない**（参照画像として機能するのみ）
   - 初期latentがランダムノイズのままだと、**完全なノイズ出力**になる

2. **初期latentの手動作成の必要性:**
   - 入力画像をエンコードして`x0`（元画像のlatent表現）を取得
   - `denoise`値に応じてノイズを混合して`x_t`を作成
   - これを`pipeline_kwargs["latents"]`に渡すことで、**真のi2i**を実現

3. **`image`引数の削除（重要）:**
```python
# nodes/samplerv2.py:567-572
# IMPORTANT:
# If we successfully initialize `latents` from the input image, do NOT also pass `image=`.
# Flux2 treats `image` as additional reference conditioning tokens; keeping it makes denoise appear
# "stuck" (0.2 and 0.8 look similar) because the reference conditioning dominates.
if flux_latent_init_ok:
    pipeline_kwargs.pop("image", None)
```

**なぜ`image`を削除するか:**
- `image`を残すと、**参照条件付けが強すぎる**
- `denoise`値（0.2 vs 0.8）による差異が**ほとんど見えなくなる**
- 初期latent（`x_t`）によるi2i効果が、参照条件付けに**上書きされる**

**標準KSamplerとの根本的違い:**
1. **手動での初期latent準備**: Flux2では必須
2. **sigmaスケジュールの明示的制御**: Flow Matching特有
3. **`image`引数の条件付き削除**: Flux2の特殊な動作に対応

---

## 4. Flux2特有の特殊実装

### 4.1 `retrieve_timesteps`のパッチ

**実装ロジック:**
```python
# nodes/samplerv2.py:615-642
if pipeline_type in ["Flux2Pipeline", "FluxPipeline"]:
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_timesteps
    import diffusers.pipelines.flux2.pipeline_flux2 as flux2_module
    
    scheduler_supports_mu = isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler)
    
    def patched_retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None, **kwargs):
        if not scheduler_supports_mu:
            kwargs.pop("mu", None)  # muパラメータを削除
        return original_retrieve_timesteps(
            scheduler, num_inference_steps, device, timesteps=timesteps, **kwargs
        )
    
    flux2_module.retrieve_timesteps = patched_retrieve_timesteps
```

**なぜパッチが必要か:**
- `retrieve_timesteps`は`mu`パラメータを要求する場合がある
- 一部のスケジューラー（非FlowMatchEulerDiscreteScheduler）は`mu`をサポートしない
- `mu`を削除することで、**互換性を確保**

### 4.2 VAE.decodeのパッチ

**実装ロジック:**
```python
# nodes/samplerv2.py:603-613
if hasattr(pipeline, "vae") and pipeline.vae is not None and hasattr(pipeline.vae, "decode"):
    original_vae_decode = pipeline.vae.decode
    def patched_decode(z, *args, **kwargs):
        return original_vae_decode(z.float(), *args, **kwargs)  # float32に強制変換
    pipeline.vae.decode = patched_decode
```

**なぜパッチが必要か:**
- Flux2のVAEは`bfloat16`や`float16`で動作することがある
- デコード時に`float32`への変換が必要
- **bfloat16/float32のバイアス不一致**を回避

### 4.3 画像サイズの調整

**実装ロジック:**
```python
# nodes/samplerv2.py:521-536
img_w, img_h = pil_cond.size

# 1024x1024を超える場合はリサイズ
if img_w * img_h > 1024 * 1024:
    pil_cond = pipeline.image_processor._resize_to_target_area(pil_cond, 1024 * 1024)
    img_w, img_h = pil_cond.size

# multiple_ofに合わせて調整
multiple_of = int(getattr(pipeline, "vae_scale_factor", 16)) * 2
if multiple_of > 0:
    img_w = (img_w // multiple_of) * multiple_of
    img_h = (img_h // multiple_of) * multiple_of
```

**なぜ調整が必要か:**
- Flux2のVAEは特定のサイズ倍数を要求
- サイズ不一致はエラーや品質低下の原因
- **パイプラインの要求に完全に準拠**

---

## 5. 非Fluxパイプラインの処理

### 5.1 標準的なi2i処理

**実装ロジック:**
```python
# nodes/samplerv2.py:575-593
elif (not is_flux_family) and isinstance(latent_image, dict) and latent_image.get("vae") is not None:
    init_latents = samples
    # dtypeとdeviceを調整
    target_dtype = None
    for attr in ("unet", "transformer"):
        m = getattr(pipeline, attr, None)
        if m is not None and hasattr(m, "dtype"):
            target_dtype = m.dtype
            break
    if target_dtype is not None and init_latents.dtype != target_dtype:
        init_latents = init_latents.to(dtype=target_dtype)
    init_latents = init_latents.to(device=generator_device)
    
    pipeline_kwargs["latents"] = init_latents
    pipeline_kwargs["strength"] = strength
```

**動作原理:**
- 入力latentをそのまま使用
- diffusersの標準`strength`パラメータで制御
- パイプラインが内部でノイズ追加を処理

### 5.2 エラーハンドリング

**実装ロジック:**
```python
# nodes/samplerv2.py:648-671
try:
    result = pipeline(**pipeline_kwargs)
except TypeError as e:
    # latents/strengthがサポートされていない場合のリトライ
    if "latents" in str(e) and "unexpected keyword argument" in str(e):
        if "latents" in pipeline_kwargs:
            del pipeline_kwargs["latents"]
        if "strength" in pipeline_kwargs:
            del pipeline_kwargs["strength"]
        result = pipeline(**pipeline_kwargs)
    # width/heightがサポートされていない場合
    elif ("width" in str(e) or "height" in str(e)) and "unexpected keyword argument" in str(e):
        pipeline_kwargs.pop("width", None)
        pipeline_kwargs.pop("height", None)
        result = pipeline(**pipeline_kwargs)
```

**なぜ必要か:**
- diffusersパイプラインは種類によってAPIが異なる
- **柔軟なエラーハンドリング**により、様々なパイプラインに対応

---

## 6. 技術的課題と解決策

### 6.1 課題1: "完全なノイズ"出力

**問題:**
- Flux2でi2iを実行すると、**完全なノイズ画像**が出力される
- `denoise`値が機能しない

**原因:**
- Flux2の`image`パラメータは**条件付けトークン**として機能
- 初期latentがランダムノイズのままだと、i2iにならない

**解決策:**
1. 入力画像をエンコードして`x0`を取得（543-544行目）
2. `denoise`値に応じてノイズを混合して`x_t`を作成（558-560行目）
3. `x_t`を`pipeline_kwargs["latents"]`に渡す（563行目）
4. **`image`引数を削除**（571-572行目）

### 6.2 課題2: Denoise値の効果が弱い

**問題:**
- `denoise=0.2`と`denoise=0.8`で**ほとんど差がない**
- 参照条件付けが強すぎる

**原因:**
- `image`引数が残っていると、参照条件付けが**denoise効果を上書き**

**解決策:**
- 初期latentが正常に準備できた場合、**`image`引数を削除**（571-572行目）
- これにより、初期latentによるi2i効果が**優先される**

### 6.3 課題3: 低denoise値で元画像に近づかない

**問題:**
- `denoise=0.2`でも、元画像から**離れた結果**になる

**原因:**
- `sigma_end`が`1/steps`（例: 0.05）の場合、**ノイズフロア**が残る

**解決策:**
- `sigma_end = 0.0`に固定（512行目）
- これにより、**完全に元画像に収束可能**

### 6.4 課題4: ステップ数の削減による品質低下

**問題:**
- 標準KSamplerの`effective_steps = steps * denoise`は、**ステップ数を削減**
- 低い`denoise`値では品質が低下

**解決策:**
- SDNQSamplerV2では**ステップ数は変更しない**（384行目: `num_inference_steps: steps`）
- `denoise`値は**初期ノイズレベル**（sigma）のみを制御
- これにより、**品質を保ちながらdenoise制御**が可能

---

## 7. 実装の整合性と設計判断

### 7.1 パイプラインタイプによる分岐

**設計判断:**
```python
# nodes/samplerv2.py:311-318
pipeline_type = type(pipeline).__name__
is_flux_family = pipeline_type in ["Flux2Pipeline", "FluxPipeline", "FluxSchnellPipeline"]
call_params = set(inspect.signature(pipeline.__call__).parameters.keys())
supports_image_arg = ("image" in call_params)
```

**なぜ必要か:**
- Flux2と非Fluxでは**処理が根本的に異なる**
- 実行時にパイプラインタイプを判定して**適切な処理を選択**

### 7.2 入力形式の柔軟性

**設計判断:**
```python
# nodes/samplerv2.py:476-488
pil_cond = None
# パターン1: pixelsが直接提供されている場合
if isinstance(latent_image, dict) and latent_image.get("pixels") is not None:
    pil_cond = _tensor_to_pil_rgb(px)
# パターン2: latentからデコードする場合
if pil_cond is None and isinstance(latent_image, dict) and latent_image.get("vae") is not None:
    pil_cond = _decode_latents_to_pil(samples)
```

**なぜ必要か:**
- ComfyUIの`SDNQVAEEncode`は`pixels`を提供（最適）
- 従来のVAEエンコードノードは`latent`のみを提供
- **両方の形式に対応**することで、柔軟性を確保

### 7.3 ステップ数の保持

**設計判断:**
```python
# nodes/samplerv2.py:384
"num_inference_steps": steps,  # denoise値で変更しない
```

**標準KSamplerとの違い:**
- 標準: `effective_steps = steps * denoise`（ステップ数削減）
- SDNQSamplerV2: `steps`をそのまま使用（ステップ数保持）

**理由:**
1. **品質の維持**: ステップ数を削減しないことで、品質を保つ
2. **denoiseの正確な制御**: 初期ノイズレベル（sigma）のみを制御
3. **Flow Matchingとの整合性**: Flow Matchingではsigmaスケジュールで制御

### 7.4 エラーハンドリングの多層防御

**設計判断:**
```python
# nodes/samplerv2.py:565-566, 592-593
except Exception as e:
    print(f"[SDNQ Sampler V2] Warning: Flux img2img latent init failed, falling back to conditioning-only: {e}")
```

**なぜ必要か:**
- Flux2の初期latent準備が失敗した場合、**フォールバック**が必要
- `image`条件付けのみで処理を続行（完全なi2iではないが、エラーを回避）

---

## 8. まとめ: 実装の完全な技術的整合性

### 8.1 標準KSamplerとの根本的違い

| 項目 | 標準KSampler | SDNQSamplerV2 |
|------|-------------|--------------|
| **アーキテクチャ** | ComfyUI内部API | diffusersパイプライン |
| **Denoise制御** | ステップ数削減 | 初期ノイズレベル調整 |
| **i2i処理** | VAEエンコード/デコード | パイプライン固有処理 |
| **Flux2対応** | なし | 完全対応（特殊実装） |
| **ステップ数** | `steps * denoise` | `steps`（変更なし） |

### 8.2 Flux2特有の実装の重要性

1. **初期latentの手動作成**: 真のi2iを実現
2. **sigma_end = 0.0**: 完全な元画像収束を可能に
3. **`image`引数の削除**: denoise効果を正しく機能させる
4. **sigmaスケジュールの明示的制御**: Flow Matchingの原理に基づく

### 8.3 設計の一貫性

- **パイプラインタイプによる分岐**: Flux2と非Fluxで適切な処理を選択
- **柔軟な入力形式**: `pixels`と`latent`の両方に対応
- **多層防御**: エラーハンドリングとフォールバック
- **数学的正確性**: Flow Matchingの原理に基づく実装

### 8.4 技術的優位性

1. **品質の維持**: ステップ数を削減しないため、品質が保たれる
2. **正確なdenoise制御**: sigmaスケジュールによる数学的に正確な制御
3. **Flux2完全対応**: Flux2の特殊な動作を完全に理解し、適切に対応
4. **柔軟性**: 様々なパイプラインタイプに対応

---

## 9. コード参照

主要な実装箇所:

- **Denoise制御（Flux2）**: `nodes/samplerv2.py:502-519`
- **初期latent準備（Flux2）**: `nodes/samplerv2.py:521-564`
- **`image`引数の削除**: `nodes/samplerv2.py:571-572`
- **Denoise制御（非Flux）**: `nodes/samplerv2.py:573-574`
- **エラーハンドリング**: `nodes/samplerv2.py:648-671`
- **VAE.decodeパッチ**: `nodes/samplerv2.py:603-613`
- **retrieve_timestepsパッチ**: `nodes/samplerv2.py:615-642`

---

**この実装により、SDNQSamplerV2は標準KSamplerを超える機能を提供し、特にFlux2モデルにおいて、正確で高品質なi2iとdenoise制御を実現しています。**

