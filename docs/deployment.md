# Deploying the project page to `jinnn.xyz`

`site/` is a **self-contained static page** (HTML/CSS/JS + local images/video). It has no build step
and no external CDN. You can host it anywhere that serves static files.

> I cannot publish to your domain from here: that needs your GitHub account, DNS console, and/or
> server credentials. The steps below are the exact operations; choose **one** option.

---

## Your domain is already on GitHub Pages

`https://jinnn.xyz` currently responds with `Server: GitHub.com` and hosts your personal site.
**Do not replace the root page** with this project. Two clean choices:

| Choice | URL | Work |
|---|---|---|
| **A. Subpath (recommended)** | `https://jinnn.xyz/mine-loco/` | Copy `site/` into the existing Pages repo as `mine-loco/`; no DNS change |
| **B. Subdomain** | `https://mine-loco.jinnn.xyz` | New repo + one `CNAME` DNS record + Pages custom domain |

### A. Subpath `jinnn.xyz/mine-loco/` (recommended)

All links in `site/` are relative, so the page works under any subpath.

```bash
# 1) clone the repo that serves jinnn.xyz (your personal Pages repo)
git clone https://github.com/luongthimydieu/luongthimydieu.github.io.git
cd luongthimydieu.github.io

# 2) copy the static page into a subfolder
mkdir -p mine-loco
rsync -av --delete /path/to/mine-loco-perception/site/ mine-loco/

# 3) commit and push (Pages rebuilds automatically)
git add mine-loco
git commit -m "Add Mine-Loco project page"
git push
```

Open `https://jinnn.xyz/mine-loco/`. If the repo/branch differs, use its actual Pages source
(Settings → Pages).

> If the personal site is a **project** Pages repo (not `<user>.github.io`), the URL is
> `https://<user>.github.io/<repo>/mine-loco/` unless a custom domain is set; with `jinnn.xyz` as the
> custom domain it becomes `https://jinnn.xyz/<repo>/mine-loco/`. Check the repo's Pages settings.

### B. Subdomain `mine-loco.jinnn.xyz`

1. Push this repository to GitHub (e.g. `luongthimydieu/mine-loco-perception`).
2. Repo → Settings → Pages → Source: **GitHub Actions**. The included workflow publishes `site/`.
3. After the first deploy, set **Custom domain** to `mine-loco.jinnn.xyz` and save.
4. At the DNS provider for `jinnn.xyz`, add:

```
Type   Name        Value
CNAME  mine-loco   luongthimydieu.github.io
```

5. Wait for DNS, then enable **Enforce HTTPS**. GitHub provisions the certificate automatically.

> If the DNS is on Cloudflare, set the record to **DNS only** (grey cloud) while GitHub verifies the
> domain; you can proxy it afterwards if you want Cloudflare caching.

### Updating

- Subpath: copy `site/` again and push the personal Pages repo.
- Subdomain: push this repository; the Pages workflow redeploys.

---

## Other hosting options

The rest of this document keeps the generic Cloudflare Pages / VPS / GitHub Pages instructions. They
are useful only if you decide to move the domain or host the page elsewhere; for `jinnn.xyz` as it is
today, use **A** or **B** above.

---

## 0. Preview locally

```bash
cd site
python -m http.server 8000
# open http://localhost:8000
```

Check that the video, figures and the visibility calculator work before deploying.

---

## Option A — Cloudflare Pages (recommended, free, HTTPS)

**Prerequisites:** the `jinnn.xyz` domain is added to Cloudflare (its nameservers point to Cloudflare).

1. Push this repository to GitHub.
2. Cloudflare Dashboard → **Workers & Pages** → **Create** → **Pages** → **Connect to Git**.
3. Select the repository. Settings:
   - **Framework preset**: `None`
   - **Build command**: leave empty (or `exit 0`)
   - **Build output directory**: `site`
4. **Save and Deploy**. Wait for the first deployment; open the `*.pages.dev` URL.
5. **Custom domains** → **Set up a custom domain**:
   - Root domain: `jinnn.xyz`
   - Optional: `www.jinnn.xyz`
6. DNS (Cloudflare manages it automatically when the zone is on Cloudflare):
   - Root: Cloudflare uses CNAME flattening; ensure no old `A`/`AAAA`/`CNAME` records for `@` remain.
   - Subdomain (`mine-loco.jinnn.xyz`): `CNAME` → `<project>.pages.dev` (Proxied).
7. **SSL/TLS** → Overview → **Full (strict)**; **Edge Certificates** → **Always Use HTTPS** on.
8. Verify:

```bash
curl -I https://jinnn.xyz
# HTTP/2 200, content-type: text/html
```

**Updating:** every `git push` to the production branch triggers a new deployment. Rollback is in
Pages → Deployments → “Rollback”.

---

## Option B — VPS + Nginx (full control)

Assumes Ubuntu 22.04, Nginx installed, and a DNS `A` record for `jinnn.xyz` pointing to the server IP.

```bash
# 1) Upload the static site (run from the repo root)
rsync -avz --delete site/ user@YOUR_SERVER:/var/www/mine-loco/

# 2) Nginx site
sudo tee /etc/nginx/sites-available/mine-loco >/dev/null <<'EOF'
server {
    listen 80;
    listen [::]:80;
    server_name jinnn.xyz www.jinnn.xyz;

    root /var/www/mine-loco;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }

    # long cache for images/video
    location ~* \.(jpg|jpeg|png|webp|mp4|css|js)$ {
        expires 7d;
        add_header Cache-Control "public, max-age=604800";
    }

    gzip on;
    gzip_types text/css application/javascript image/svg+xml;
}
EOF

sudo ln -s /etc/nginx/sites-available/mine-loco /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 3) HTTPS
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d jinnn.xyz -d www.jinnn.xyz
```

Open ports 80/443 in the firewall. **Updating:** rerun the `rsync` command; Nginx serves the new files
immediately.

**Rollback:** keep releases and switch a symlink:

```bash
sudo mv /var/www/mine-loco /var/www/mine-loco-$(date +%F)
# re-upload an older copy, or keep /var/www/mine-loco-prev and repoint
```

---

## Option C — GitHub Pages + custom domain

The repository includes `.github/workflows/pages.yml`, which publishes `site/` on every push to `main`.

1. GitHub repo → **Settings** → **Pages** → **Build and deployment** → Source: **GitHub Actions**.
2. Push to `main`; the workflow uploads `site/` as a Pages artifact.
3. **Custom domain** → enter `jinnn.xyz` → Save. GitHub creates a `CNAME` file in the Pages branch.
4. DNS at your provider:
   - **Root domain** `jinnn.xyz`: `A` records to
     `185.199.108.153`, `185.199.109.153`, `185.199.110.153`, `185.199.111.153`
     (and optional `AAAA` to `2606:50c0:8000::153`, `2606:50c0:8001::153`,
     `2606:50c0:8002::153`, `2606:50c0:8003::153`).
   - **Subdomain** `mine-loco.jinnn.xyz`: `CNAME` → `<your-user>.github.io`.
5. Wait for DNS, then enable **Enforce HTTPS** in Pages settings.

If you use Cloudflare DNS with GitHub Pages, set the records to **DNS only** (grey cloud) during
domain verification; you can turn the proxy on after HTTPS is issued.

---

## Which option should I choose?

| Situation | Choose |
|---|---|
| Domain already on Cloudflare, want zero server maintenance | **A. Cloudflare Pages** |
| Already have a VPS and want full control | **B. Nginx** |
| Want everything inside GitHub | **C. GitHub Pages** |

All three serve the same `site/` directory; switching later is just a re-upload/reconnect.

---

## Post-deploy checklist

- [ ] `https://jinnn.xyz` returns 200 over HTTPS
- [ ] figures and `assets/demo.mp4` load (no 404 in DevTools → Network)
- [ ] the visibility calculator updates when sliders move
- [ ] mobile layout at 390 px width is readable
- [ ] `robots.txt` / social preview (optional)
- [ ] no personal data, phone numbers, or competition documents are present
- [ ] GitHub repo link in the nav points to the real repository

## If you want me to walk through it live

Tell me which option you use and provide (in chat, not committed):
- the hosting platform (Cloudflare / VPS / GitHub Pages),
- the GitHub repo URL,
- the DNS provider for `jinnn.xyz`,
- for a VPS: the SSH host/user (do **not** paste a password; use a key).

Then I can give you the exact commands for that specific setup.
