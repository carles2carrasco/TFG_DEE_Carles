<template>
  <ion-page>
    <ion-header :translucent="true">
      <ion-toolbar>
        <ion-title id="title">Following You</ion-title>
      </ion-toolbar>
    </ion-header>

    <ion-content :fullscreen="true">
      <ion-header collapse="condense">
        <ion-toolbar>
          <ion-title size="large">Following You</ion-title>
        </ion-toolbar>
      </ion-header>

      <div id="container">
        <p id="locationInfo">Coordinates</p>
        <p v-if="coordenadas">Latitude: {{ coordenadas.latitude }}</p>
        <p v-if="coordenadas">Longitude: {{ coordenadas.longitude }}</p>
      </div>
      <ion-button class="botonLocalizador" @click="obtenerCoordenadas">Get Simple Location</ion-button>
    </ion-content>
  </ion-page>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import mqtt from 'mqtt';

const coordenadas = ref<{ latitude: number; longitude: number } | null>(null);

// Conectar al broker MQTT
const client = mqtt.connect('ws://broker.hivemq.com:8000/mqtt');

client.on('connect', () => {
  console.log('Conectado al broker MQTT');
});

const publicarCoordenadas = (latitude: number, longitude: number) => {
  const message = JSON.stringify({ latitude, longitude });
  client.publish('CarlesDEECoordenadas', message, {}, (error) => {
    if (error) {
      console.error('Error al publicar en el topic:', error);
    } else {
      console.log('Coordenadas publicadas:', message);
    }
  });
};

const obtenerCoordenadas = () => {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      position => {
        coordenadas.value = {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude
        };
        publicarCoordenadas(position.coords.latitude, position.coords.longitude);
      },
      error => {
        alert(`Error al obtener las coordenadas: ${error.message}`);
      }
    );
  } else {
    alert("El navegador no soporta la geolocalización.");
  }
};
</script>

<style scoped>
#container {
  text-align: center;
  position: absolute;
  left: 0;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
}

#container strong {
  font-size: 20px;
  line-height: 26px;
}

#container p {
  font-size: 16px;
  line-height: 16px;
  color: #8c8c8c;
  margin: 0;
}

#title {
  text-align: center;
}

.botonLocalizador {
  display: block;
  margin: 10px auto;
  width: 80%;
}
</style>