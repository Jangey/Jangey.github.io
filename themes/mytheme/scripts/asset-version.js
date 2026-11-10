'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

// Cloudflare caches theme assets for hours; a content hash in the URL makes each deploy fetch fresh copies.
hexo.extend.helper.register('asset_version', function (file) {
  const content = fs.readFileSync(path.join(hexo.theme_dir, 'source', file));
  const hash = crypto.createHash('md5').update(content).digest('hex').slice(0, 8);
  return `${this.url_for(file)}?v=${hash}`;
});
