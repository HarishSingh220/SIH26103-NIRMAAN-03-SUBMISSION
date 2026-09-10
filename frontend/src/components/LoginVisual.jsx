import indiaMap from "../assets/india.svg";

function LoginVisual() {
  return (
    <section className="login-visual">

      <div className="brand-block">
        <div className="brand-logo">N</div>

        <div>
          <h1>NIRMAAN</h1>
          <p>Project Monitoring</p>
        </div>
      </div>

      <div className="hero-text">
        <span>TRANSPARENT PROJECTS.</span>
        <strong>A STRONGER INDIA.</strong>
      </div>

      <div className="india-network">

        <div className="india-glow"></div>

        <div className="india-map">

          <img
            src={indiaMap}
            alt=""
            className="india-svg"
          />

          <span className="map-node node-a"></span>
          <span className="map-node node-b"></span>
          <span className="map-node node-c"></span>
          <span className="map-node node-d"></span>
          <span className="map-node node-e"></span>
          <span className="map-node node-f"></span>
          <span className="map-node node-g"></span>
          <span className="map-node node-h"></span>

        </div>

        <span className="orbit-dot orbit-1"></span>
        <span className="orbit-dot orbit-2"></span>
        <span className="orbit-dot orbit-3"></span>

      </div>

      <div className="hero-footer">

        <span className="saffron"></span>
        <span className="white"></span>
        <span className="green"></span>

        <p>
          BETTER PLANNING&nbsp;&nbsp; / &nbsp;&nbsp;
          SMARTER IMPLEMENTATION&nbsp;&nbsp; / &nbsp;&nbsp;
          BRIGHTER TOMORROW
        </p>

      </div>

    </section>
  );
}

export default LoginVisual;