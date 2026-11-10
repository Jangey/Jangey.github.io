'use strict';

hexo.extend.filter.register('after_post_render', data => {
  data.content = data.content.replace(/<(img|iframe)\b(?![^>]*\bloading=)/gi, '<$1 loading="lazy"');
  return data;
});
