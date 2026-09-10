import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import "./Home.css";
import "../styles/nirmaan-sections.css";
import Navbar from "../components/Navbar";
import SectorMinistrySection from "../components/SectorMinistry/SectorMinistrySection";
import StateWiseSection from "../components/StateWise/StateWiseSection";
import HighValueSection from "../components/HighValue/HighValueSection";
import data from "../data/projects-summary.json";
import { formatCount, formatLakhCr } from "../utils/format";

import carousel01 from "../assets/home-carousel-01.png";
import carousel02 from "../assets/home-carousel-02.png";
import carousel03 from "../assets/home-carousel-03.png";
import carousel04 from "../assets/home-carousel-04.png";
import carousel05 from "../assets/home-carousel-05.png";

const slides = [
  { image: carousel05, alt: "Infrastructure construction site" },
  { image: carousel01, alt: "Telecommunications infrastructure towers" },
  { image: carousel02, alt: "Port and shipping infrastructure at sunset" },
  { image: carousel03, alt: "Mining and heavy infrastructure operations" },
  { image: carousel04, alt: "Power generation infrastructure" },
];

const AUTOPLAY_MS = 4500;

function HomeHero() {
  const [activeSlide, setActiveSlide] = useState(0);
  const [landingAnimationDone, setLandingAnimationDone] = useState(false);
  const autoplayRef = useRef(null);

  useEffect(() => {
    const timer = window.setTimeout(() => setLandingAnimationDone(true), 1550);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    autoplayRef.current = window.setInterval(() => {
      setActiveSlide((current) => (current + 1) % slides.length);
    }, AUTOPLAY_MS);

    return () => window.clearInterval(autoplayRef.current);
  }, []);

  const goToSlide = (index) => {
    setActiveSlide(index);
    window.clearInterval(autoplayRef.current);
    autoplayRef.current = window.setInterval(() => {
      setActiveSlide((current) => (current + 1) % slides.length);
    }, AUTOPLAY_MS);
  };

  const goToPrevious = () => {
    goToSlide((activeSlide - 1 + slides.length) % slides.length);
  };

  const goToNext = () => {
    goToSlide((activeSlide + 1) % slides.length);
  };

  return (
    <section className="hero" aria-label="NIRMAAN introduction and infrastructure carousel">
      <div className={`hero-carousel ${landingAnimationDone ? "landing-complete" : "landing-active"}`}>
        {slides.map((slide, index) => (
          <div
            className={`hero-slide ${index === activeSlide ? "is-active" : ""}`}
            key={slide.image}
            aria-hidden={index !== activeSlide}
          >
            <img src={slide.image} alt={slide.alt} />
          </div>
        ))}
        <div className="hero-image-shade" aria-hidden="true" />
        <div className="hero-image-grain" aria-hidden="true" />
      </div>

      <div className="hero-content">
        <span className="hero-tricolor" aria-hidden="true">
          <span></span>
          <span></span>
          <span></span>
        </span>

        <p className="hero-eyebrow">
          Infrastructure &amp; Project Monitoring Division
        </p>

        <h1>
          Tracking India&rsquo;s infrastructure, <span>project by project.</span>
        </h1>

        <p>
          NIRMAAN consolidates cost, schedule and expenditure data for
          thousands of infrastructure projects across sectors, ministries
          and states &mdash; giving planners and citizens one transparent
          view of where public investment is going.
        </p>

        <div className="hero-actions">
          <Link to="/project-overview" className="btn btn-primary">
            View Project Overview
          </Link>
          <Link to="/risk-analysis" className="btn btn-secondary">
            Explore Risk Analysis
          </Link>
        </div>
      </div>

      <div className="hero-controls" aria-label="Carousel controls">
        <button type="button" className="hero-arrow" onClick={goToPrevious} aria-label="Previous slide">
          <span aria-hidden="true">←</span>
        </button>
        <div className="hero-dots">
          {slides.map((slide, index) => (
            <button
              type="button"
              key={slide.image}
              className={`hero-dot ${index === activeSlide ? "is-active" : ""}`}
              onClick={() => goToSlide(index)}
              aria-label={`Go to slide ${index + 1}`}
              aria-current={index === activeSlide ? "true" : undefined}
            />
          ))}
        </div>
        <button type="button" className="hero-arrow" onClick={goToNext} aria-label="Next slide">
          <span aria-hidden="true">→</span>
        </button>
      </div>

      <div className="hero-progress" aria-hidden="true">
        <span key={activeSlide} />
      </div>
    </section>
  );
}

function Home() {
  const { totals } = data;

  return (
    <>
      <Navbar />

      <div className="home">
        <HomeHero />

        <div className="home-header">
          <div>
            <p className="eyebrow">IPMD • PROJECT MONITORING</p>

            <h1>Project Monitoring Dashboard</h1>

            <p className="subtitle">
              A centralized overview of ongoing infrastructure projects
              across sectors, ministries and states.
            </p>
          </div>

          
        </div>

        <div className="stats">
          <div className="stat-card">
            <span>Total Projects</span>
            <strong>{formatCount(totals.projects)}</strong>
            <small>Across all sectors</small>
          </div>

          <div className="stat-card">
            <span>Total Project Value</span>
            <strong>{formatLakhCr(totals.originalCost)}</strong>
            <small>Original sanctioned cost</small>
          </div>

          <div className="stat-card">
            <span>Cumulative Expenditure</span>
            <strong>{formatLakhCr(totals.expenditure)}</strong>
            <small>Spent to date</small>
          </div>

          <div className="stat-card">
            <span>States &amp; UTs Covered</span>
            <strong>{formatCount(totals.states)}</strong>
            <small>Plus multi-state projects</small>
          </div>
        </div>

        <SectorMinistrySection />
        <HighValueSection />
        <StateWiseSection />
      </div>
    </>
  );
}

export default Home;
