window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // TODO: Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

$(document).ready(function () {
  // Init bulma-carousel (NOT bulma-slider)
  const instances = bulmaCarousel.attach('#results-carousel', {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: false,      // we control timing via video, not a fixed timer
    navigation: true,
    pagination: true
  });
  const carousel = Array.isArray(instances) ? instances[0] : instances;

  const videos = document.querySelectorAll('#results-carousel video');

  // Ensure policy-friendly autoplay
  videos.forEach(v => {
    v.muted = true;                     // required for autoplay
    v.setAttribute('playsinline', '');  // iOS
    v.removeAttribute('loop');          // we need 'ended' to fire
  });

  function pauseAndResetAll() {
    videos.forEach(v => { v.pause(); if (!v.closest('.is-active')) v.currentTime = 0; });
  }

  function getActiveVideo() {
    // bulma-carousel marks the active slide with .is-active
    return document.querySelector('#results-carousel .is-active video');
  }

  function playActiveVideo() {
    const v = getActiveVideo();
    if (!v) return;
    // If metadata not ready, wait for it once, then play.
    if (v.readyState >= 1) v.play().catch(() => {});
    else v.addEventListener('loadedmetadata', () => v.play().catch(() => {}), { once: true });
  }

  // Advance when the current slide's video ends
  videos.forEach(v => v.addEventListener('ended', () => carousel?.next && carousel.next()));

  // When slide changes, switch the playing video
  if (carousel?.on) {
    carousel.on('before:show', () => pauseAndResetAll());
    carousel.on('after:show', () => playActiveVideo());
  }

  // Start on load
  pauseAndResetAll();
  playActiveVideo();

  // OPTIONAL: pause auto-advance while hovering
  const wrap = document.getElementById('results-carousel');
  let hoverBlocked = false;
  if (wrap) {
    wrap.addEventListener('mouseenter', () => { hoverBlocked = true; });
    wrap.addEventListener('mouseleave', () => { hoverBlocked = false; });
    videos.forEach(v => v.addEventListener('ended', () => { if (!hoverBlocked) carousel.next(); }));
  }

  // Keep your visibility-based pause if you like:
  setupVideoCarouselAutoplay?.();
});
